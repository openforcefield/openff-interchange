"""An object for storing, manipulating, and converting molecular mechanics data."""
import copy
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
from openff.models.models import DefaultModel
from openff.models.types import ArrayQuantity, QuantityEncoder, custom_quantity_encoder
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit
from openff.utilities.utilities import has_package, requires_package
from pydantic import Field, validator

from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.components.potentials import Collection
from openff.interchange.exceptions import (
    InvalidBoxError,
    InvalidTopologyError,
    MissingParameterHandlerError,
    MissingPositionsError,
    UnsupportedCombinationError,
    UnsupportedExportError,
)
from openff.interchange.models import PotentialKey, TopologyKey
from openff.interchange.warnings import InterchangeDeprecationWarning

if TYPE_CHECKING:
    from openff.interchange.components.foyer import (
        FoyerElectrostaticsHandler,
        FoyerVDWHandler,
    )
    from openff.interchange.smirnoff._nonbonded import (
        SMIRNOFFElectrostaticsCollection,
        SMIRNOFFvdWCollection,
    )
    from openff.interchange.smirnoff._valence import (
        SMIRNOFFAngleCollection,
        SMIRNOFFBondCollection,
        SMIRNOFFConstraintCollection,
        SMIRNOFFImproperTorsionCollection,
        SMIRNOFFProperTorsionCollection,
    )
    from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection

    if has_package("foyer"):
        from foyer.forcefield import Forcefield as FoyerForcefield
    if has_package("nglview"):
        import nglview


def _sanitize(o):
    # `BaseModel.json()` assumes that all keys and values in dicts are JSON-serializable, which is a problem
    # for the mapping dicts `key_map` and `potentials`.
    if isinstance(o, dict):
        return {_sanitize(k): _sanitize(v) for k, v in o.items()}
    elif isinstance(o, (PotentialKey, TopologyKey)):
        return o.json()
    elif isinstance(o, unit.Quantity):
        return custom_quantity_encoder(o)
    return o


class TopologyEncoder(json.JSONEncoder):
    """Custom encoder for `Topology` objects."""

    def default(self, obj: "Topology"):  # noqa
        _topology = copy.deepcopy(obj)
        for molecule in _topology.molecules:
            molecule._conformers = None

        return _topology.to_json()


def interchange_dumps(v, *, default):
    """Dump an Interchange to JSON after converting to compatible types."""
    return json.dumps(
        {
            "positions": QuantityEncoder().default(v["positions"]),
            "box": QuantityEncoder().default(v["box"]),
            "topology": TopologyEncoder().default(v["topology"]),
            "collections": {
                "Bonds": json.dumps(
                    _sanitize(v["collections"]["Bonds"]),
                    default=default,
                ),
            },
        },
        default=default,
    )


def interchange_loader(data: str) -> dict:
    """Load a JSON representation of an Interchange object."""
    tmp: Dict = {
        "positions": None,
        "velocities": None,
        "box": None,
        "topology": None,
        "collections": {},
    }
    for key, val in json.loads(data).items():
        if val is None:
            continue
        if key == "positions":
            tmp["positions"] = unit.Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "velocities":
            tmp["velocities"] = unit.Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "box":
            tmp["box"] = unit.Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "topology":
            tmp["topology"] = Topology.from_json(val)

    return tmp


class Interchange(DefaultModel):
    """
    A object for storing, manipulating, and converting molecular mechanics data.

    .. warning :: This object is in an early and experimental state and unsuitable for production.
    .. warning :: This API is experimental and subject to change.
    """

    collections: Dict[str, Collection] = Field(dict())
    topology: Topology = Field(None)
    mdconfig: MDConfig = Field(None)
    box: ArrayQuantity["nanometer"] = Field(None)
    positions: ArrayQuantity["nanometer"] = Field(None)
    velocities: ArrayQuantity["nanometer/picosecond"] = Field(None)

    class Config:
        """Custom Pydantic-facing configuration for the Interchange class."""

        json_loads = interchange_loader
        json_dumps = interchange_dumps
        validate_assignment = True
        arbitrary_types_allowed = True

    @validator("box", allow_reuse=True)
    def validate_box(cls, value):
        if value is None:
            return value
        first_pass = ArrayQuantity.validate_type(value)
        as_2d = np.atleast_2d(first_pass)
        if as_2d.shape == (3, 3):
            box = as_2d
        elif as_2d.shape == (1, 3):
            box = as_2d * np.eye(3)
        else:
            raise InvalidBoxError(
                f"Failed to convert value {value} to 3x3 box vectors. Please file an issue if you think this "
                "input should be supported and the failure is an error.",
            )

        return box

    @validator("topology", pre=True)
    def validate_topology(cls, value):
        if value is None:
            return None
        if isinstance(value, Topology):
            try:
                return Topology(other=value)
            except Exception as exception:
                # Topology cannot roundtrip with simple molecules
                for molecule in value.molecules:
                    if molecule.__class__.__name__ == "_SimpleMolecule":
                        return value
                raise exception
        elif isinstance(value, list):
            return Topology.from_molecules(value)
        elif value.__class__.__name__ == "_OFFBioTop":
            raise InvalidTopologyError("_OFFBioTop is no longer supported")
        else:
            raise InvalidTopologyError(
                "Could not process topology argument, expected openff.toolkit.topology.Topology. "
                f"Found object of type {type(value)}.",
            )

    def _infer_positions(self) -> Optional[ArrayQuantity]:
        """
        Attempt to set Interchange.positions based on conformers in molecules in the topology.

        If _any_ molecule lacks conformers, return None.
        If _all_ molecules have conformers, return an array of shape (self.topology.n_atoms, 3)
        generated by concatenating the positions of each molecule, using only the 0th conformer.
        """
        for molecule in self.topology.molecules:
            if molecule.n_conformers == 0:
                # if _any_ molecule lacks conformers, break out immediately
                return None

        return np.concatenate(
            [molecule.conformers[0] for molecule in self.topology.molecules],
        )

    @classmethod
    def from_smirnoff(
        cls,
        force_field: ForceField,
        topology: Union[Topology, List[Molecule]],
        box=None,
        positions=None,
        charge_from_molecules: Optional[List[Molecule]] = None,
        partial_bond_orders_from_molecules: Optional[List[Molecule]] = None,
        allow_nonintegral_charges: bool = False,
    ) -> "Interchange":
        """
        Create a new object by parameterizing a topology with a SMIRNOFF force field.

        Parameters
        ----------
        force_field : `openff.toolkit.ForceField`
            The force field to parameterize the topology with.
        topology : `openff.toolkit.topology.Topology` or `List[openff.toolkit.topology.Molecule]`
            The topology to parameterize, or a list of molecules to construct a
            topology from and parameterize.
        box : `openff.unit.Quantity`, optional
            The box vectors associated with the ``Interchange``. If ``None``,
            box vectors are taken from the topology, if present.
        positions : `openff.unit.Quantity`, optional
            The positions associated with atoms in the input topology. If ``None``,
            positions are taken from the molecules in topology, if present on all molecules.
        charge_from_molecules : `List[openff.toolkit.molecule.Molecule]`, optional
            If specified, partial charges will be taken from the given molecules
            instead of being determined by the force field.
        partial_bond_orders_from_molecules : List[openff.toolkit.molecule.Molecule], optional
            If specified, partial bond orders will be taken from the given molecules
            instead of being determined by the force field.
        allow_nonintegral_charges : bool, optional, default=False
            If True, allow molecules to have approximately non-integral charges.

        Notes
        -----
        If the `Molecule` objects in the `topology` argument each contain conformers, the returned `Interchange` object
        will have its positions set via concatenating the 0th conformer of each `Molecule`.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        OpenFF 2.0.0 "Sage"

        .. code-block:: pycon

            >>> from openff.interchange import Interchange
            >>> from openff.toolkit.topology import Molecule
            >>> from openff.toolkit.typing.engines.smirnoff import ForceField
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> sage = ForceField("openff-2.0.0.offxml")
            >>> interchange = Interchange.from_smirnoff(topology=[mol], force_field=sage)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        from openff.interchange.smirnoff._create import _create_interchange

        return _create_interchange(
            force_field=force_field,
            topology=topology,
            box=box,
            positions=positions,
            charge_from_molecules=charge_from_molecules,
            partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
            allow_nonintegral_charges=allow_nonintegral_charges,
        )

    def visualize(self, backend: str = "nglview"):
        """
        Visualize this Interchange.

        This currently only uses NGLview. Other engines may be added in the future.

        Parameters
        ----------
        backend : str, default="nglview"
            The backend to use for visualization. Currently only "nglview" is supported.

        Returns
        -------
        widget : nglview.NGLWidget
            The NGLWidget containing the visualization.

        """
        if backend == "nglview":
            return self._visualize_nglview()
        else:
            raise UnsupportedExportError

    @requires_package("nglview")
    def _visualize_nglview(self) -> "nglview.NGLWidget":
        """Visualize the system using NGLView via a PDB file."""
        import nglview

        try:
            self.to_pdb("_tmp_pdb_file.pdb", writer="openmm")
        except MissingPositionsError as error:
            raise MissingPositionsError(
                "Cannot visualize system without positions.",
            ) from error
        return nglview.show_file("_tmp_pdb_file.pdb")

    def to_gro(self, file_path: Union[Path, str], writer="internal", decimal: int = 3):
        """Export this Interchange object to a .gro file."""
        # TODO: Enum-style class for handling writer arg?
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.interchange.interop.internal.gromacs import to_gro

            to_gro(self, file_path, decimal=decimal)

        else:
            raise UnsupportedExportError

    def to_top(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to a .top file."""
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.interchange.interop.internal.gromacs import to_top

            to_top(self, file_path)

        else:
            raise UnsupportedExportError

    def to_lammps(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to a LAMMPS data file."""
        if writer == "internal":
            from openff.interchange.interop.internal.lammps import to_lammps

            to_lammps(self, file_path)
        else:
            raise UnsupportedExportError

    def to_openmm(self, combine_nonbonded_forces: bool = True):
        """Export this Interchange to an OpenMM System."""
        from openff.interchange.interop.openmm import to_openmm as to_openmm_

        return to_openmm_(self, combine_nonbonded_forces=combine_nonbonded_forces)

    def to_openmm_topology(
        self,
        ensure_unique_atom_names: Union[str, bool] = "residues",
    ):
        """Export components of this Interchange to an OpenMM Topology."""
        from openff.interchange.interop.openmm._topology import to_openmm_topology

        return to_openmm_topology(
            self,
            ensure_unique_atom_names=ensure_unique_atom_names,
        )

    def to_prmtop(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to an Amber .prmtop file."""
        if writer == "internal":
            from openff.interchange.interop.amber.export import to_prmtop

            to_prmtop(self, file_path)

        elif writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        else:
            raise UnsupportedExportError

    def to_pdb(self, file_path: Union[Path, str], writer="openmm"):
        """Export this Interchange to a .pdb file."""
        if self.positions is None:
            raise MissingPositionsError(
                "Positions are required to write a `.pdb` file but found None.",
            )

        if writer == "openmm":
            from openff.interchange.interop.openmm import _to_pdb

            _topology = Topology(other=self.topology)
            _topology.box_vectors = self.box

            _to_pdb(file_path, _topology, self.positions)
        else:
            raise UnsupportedExportError

    def to_psf(self, file_path: Union[Path, str]):
        """Export this Interchange to a CHARMM-style .psf file."""
        raise UnsupportedExportError

    def to_crd(self, file_path: Union[Path, str]):
        """Export this Interchange to a CHARMM-style .crd file."""
        raise UnsupportedExportError

    def to_inpcrd(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to an Amber .inpcrd file."""
        if writer == "internal":
            from openff.interchange.interop.amber.export import to_inpcrd

            to_inpcrd(self, file_path)

        elif writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        else:
            raise UnsupportedExportError

    def _to_parmed(self):
        """Export this Interchange to a ParmEd Structure."""
        from openff.interchange.interop.parmed import _to_parmed

        return _to_parmed(self)

    @classmethod
    def _from_parmed(cls, structure):
        from openff.interchange.interop.parmed import _from_parmed

        return _from_parmed(cls, structure)

    @classmethod
    @requires_package("foyer")
    def from_foyer(
        cls,
        force_field: "FoyerForcefield",
        topology: "Topology",
        **kwargs,
    ) -> "Interchange":
        """
        Create an Interchange object from a Foyer force field and an OpenFF topology.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        the Foyer implementation of OPLS-AA

        .. code-block:: pycon

            >>> from openff.interchange import Interchange
            >>> from openff.toolkit.topology import Molecule, Topology
            >>> from foyer import Forcefield
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> top = Topology.from_molecules([mol])
            >>> oplsaa = Forcefield(name="oplsaa")
            >>> interchange = Interchange.from_foyer(topology=top, force_field=oplsaa)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        from openff.interchange.components.foyer import get_handlers_callable

        system = cls()
        system.topology = topology

        # This block is from a mega merge, unclear if it's still needed
        for name, Handler in get_handlers_callable().items():
            if name == "Electrostatics":
                handler = Handler(scale_14=force_field.coulomb14scale)
            if name == "vdW":
                handler = Handler(scale_14=force_field.lj14scale)
            else:
                handler = Handler()

            system.collections[name] = handler

        vdw_handler: FoyerVDWHandler = system["vdW"]  # type: ignore[assignment]

        vdw_handler.store_matches(force_field, topology=system.topology)
        vdw_handler.store_potentials(force_field=force_field)

        atom_slots = vdw_handler.key_map

        electrostatics: FoyerElectrostaticsHandler = system["Electrostatics"]  # type: ignore[assignment]
        electrostatics.store_charges(
            atom_slots=atom_slots,
            force_field=force_field,
        )

        vdw_handler.scale_14 = force_field.lj14scale  # type: ignore[assignment]
        electrostatics.scale_14 = force_field.coulomb14scale  # type: ignore[assignment]

        for name, handler in system.collections.items():
            if name not in ["vdW", "Electrostatics"]:
                handler.store_matches(atom_slots, topology=system.topology)
                handler.store_potentials(force_field)

        # FIXME: Populate .mdconfig, but only after a reasonable number of state mutations have been tested

        charges = electrostatics.charges

        for molecule in system.topology.molecules:
            molecule_charges = [
                charges[TopologyKey(atom_indices=(system.topology.atom_index(atom),))].m  # type: ignore[attr-defined]
                for atom in molecule.atoms
            ]
            molecule.partial_charges = unit.Quantity(
                molecule_charges,
                unit.elementary_charge,
            )

        system.collections["vdW"] = vdw_handler
        system.collections["Electrostatics"] = electrostatics

        return system

    @classmethod
    @requires_package("intermol")
    def from_gromacs(
        cls,
        topology_file: Union[Path, str],
        gro_file: Union[Path, str],
        reader="intermol",
    ) -> "Interchange":
        """
        Create an Interchange object from GROMACS files.

        """
        from intermol.gromacs.gromacs_parser import GromacsParser

        from openff.interchange.interop.intermol import from_intermol_system

        intermol_system = GromacsParser(topology_file, gro_file).read()
        via_intermol = from_intermol_system(intermol_system)

        if reader == "intermol":
            return via_intermol

        elif reader == "internal":
            from openff.interchange.interop.internal.gromacs import (
                _read_box,
                _read_coordinates,
                from_top,
            )

            via_internal = from_top(topology_file, gro_file)

            via_internal.positions = _read_coordinates(gro_file)
            via_internal.box = _read_box(gro_file)
            for key in via_intermol.collections:
                if key not in [
                    "Bonds",
                    "Angles",
                    "ProperTorsions",
                    "ImproperTorsions",
                    "vdW",
                    "Electrostatics",
                ]:
                    raise Exception(f"Found unexpected handler with name {key}")

            return via_internal

        else:
            raise Exception(f"Reader {reader} is not implemented.")

    def _get_parameters(self, handler_name: str, atom_indices: Tuple[int]) -> Dict:
        """
        Get parameter values of a specific potential.

        Here, parameters are expected to be uniquely dfined by the name of
        its associated handler and a tuple of atom indices.

        Note: This method only checks for equality of atom indices and will likely fail on complex cases
        involved layered parameters with multiple topology keys sharing identical atom indices.
        """
        for handler in self.collections:
            if handler == handler_name:
                return self[handler_name]._get_parameters(atom_indices=atom_indices)
        raise MissingParameterHandlerError(
            f"Could not find parameter handler of name {handler_name}",
        )

    def __getattr__(self, attr: str):
        if attr == "handlers":
            warnings.warn(
                "The `handlers` attribute is deprecated. Use `collections` instead.",
                InterchangeDeprecationWarning,
            )
            return self.collections
        else:
            return super().__getattribute__(attr)

    @overload
    def __getitem__(self, item: Literal["Bonds"]) -> "SMIRNOFFBondCollection":
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["Constraints"],
    ) -> "SMIRNOFFConstraintCollection":
        ...

    @overload
    def __getitem__(self, item: Literal["Angles"]) -> "SMIRNOFFAngleCollection":
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["vdW"],
    ) -> Union["SMIRNOFFvdWCollection", "FoyerVDWHandler"]:
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["ProperTorsions"],
    ) -> "SMIRNOFFProperTorsionCollection":
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["ImproperTorsions"],
    ) -> "SMIRNOFFImproperTorsionCollection":
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["VirtualSites"],
    ) -> "SMIRNOFFVirtualSiteCollection":
        ...

    @overload
    def __getitem__(
        self,
        item: Literal["Electrostatics"],
    ) -> Union["SMIRNOFFElectrostaticsCollection", "FoyerElectrostaticsHandler"]:
        ...

    @overload
    def __getitem__(self, item: str) -> Collection:
        ...

    def __getitem__(self, item: str):  # noqa
        """Syntax sugar for looking up collections or other components."""
        if type(item) != str:
            raise LookupError(
                "Only str arguments can be currently be used for lookups.\n"
                f"Found item {item} of type {type(item)}",
            )
        if item == "positions":
            return self.positions
        elif item in {"box", "box_vectors"}:
            return self.box
        elif item in self.collections:
            return self.collections[item]
        else:
            raise LookupError(
                f"Could not find component {item}. This object has the following "
                f"collections registered:\n\t{[*self.collections.keys()]}",
            )

    def __add__(self, other):
        """Combine two Interchange objects. This method is unstable and likely unsafe."""
        from openff.interchange.components.toolkit import _combine_topologies

        warnings.warn(
            "Interchange object combination is experimental and likely to produce "
            "strange results. Any workflow using this method is not guaranteed to "
            "be suitable for production. Use with extreme caution and thoroughly "
            "validate results!",
        )

        self_copy = copy.deepcopy(self)

        self_copy.topology = _combine_topologies(self.topology, other.topology)
        atom_offset = self.topology.n_atoms

        for handler_name, handler in other.collections.items():
            # TODO: Actually specify behavior in this case
            try:
                self_handler = self_copy.collections[handler_name]
            except KeyError:
                self_copy.collections[handler_name] = handler
                warnings.warn(
                    f"'other' Interchange object has handler with name {handler_name} not "
                    f"found in 'self,' but it has now been added.",
                )
                continue

            for top_key, pot_key in handler.key_map.items():
                new_atom_indices = tuple(
                    idx + atom_offset for idx in top_key.atom_indices
                )
                new_top_key = top_key.__class__(**top_key.dict())
                try:
                    new_top_key.atom_indices = new_atom_indices
                except ValueError:
                    assert len(new_atom_indices) == 1
                    new_top_key.this_atom_index = new_atom_indices[0]

                self_handler.key_map.update({new_top_key: pot_key})
                if handler_name == "Constraints":
                    self_handler.constraints.update(
                        {pot_key: handler.constraints[pot_key]},
                    )
                else:
                    self_handler.potentials.update(
                        {pot_key: handler.potentials[pot_key]},
                    )

            self_copy.collections[handler_name] = self_handler

        if self_copy.positions is not None and other.positions is not None:
            new_positions = np.vstack([self_copy.positions, other.positions])
            self_copy.positions = new_positions
        else:
            warnings.warn(
                "Setting positions to None because one or both objects added together were missing positions.",
            )
            self_copy.positions = None

        if not np.all(self_copy.box == other.box):
            raise UnsupportedCombinationError(
                "Combination with unequal box vectors is not curretnly supported",
            )

        return self_copy

    def __repr__(self) -> str:
        periodic = self.box is not None
        n_atoms = self.topology.n_atoms
        return (
            f"Interchange with {len(self.collections)} collections, "
            f"{'' if periodic else 'non-'}periodic topology with {n_atoms} atoms."
        )
