"""An object for storing, manipulating, and converting molecular mechanics data."""
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.topology.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.utilities.utilities import has_package, requires_package
from pydantic import Field, validator

from openff.interchange.components.potentials import PotentialHandler
from openff.interchange.components.toolkit import _check_electrostatics_handlers
from openff.interchange.exceptions import (
    InternalInconsistencyError,
    InvalidBoxError,
    InvalidTopologyError,
    MissingParameterHandlerError,
    MissingPositionsError,
    SMIRNOFFHandlersNotImplementedError,
    UnsupportedCombinationError,
    UnsupportedExportError,
)
from openff.interchange.models import DefaultModel
from openff.interchange.types import ArrayQuantity

if TYPE_CHECKING:
    if has_package("foyer"):
        from foyer.forcefield import Forcefield as FoyerForcefield
    if has_package("nglview"):
        import nglview

_SUPPORTED_SMIRNOFF_HANDLERS = {
    "Constraints",
    "Bonds",
    "Angles",
    "ProperTorsions",
    "ImproperTorsions",
    "vdW",
    "Electrostatics",
    "LibraryCharges",
    "ChargeIncrementModel",
    "VirtualSites",
}


class Interchange(DefaultModel):
    """
    A object for storing, manipulating, and converting molecular mechanics data.

    .. warning :: This object is in an early and experimental state and unsuitable for production.
    .. warning :: This API is experimental and subject to change.
    """

    class _InnerSystem(DefaultModel):
        """Inner representation of Interchange components."""

        # TODO: Ensure these fields are hidden from the user as intended
        handlers: Dict[str, PotentialHandler] = dict()
        topology: Union[Topology, List, None] = Field(None)
        box: ArrayQuantity["nanometer"] = Field(None)  # type: ignore
        positions: ArrayQuantity["nanometer"] = Field(None)  # type: ignore
        velocities: ArrayQuantity["nanometer/picosecond"] = Field(None)  # type: ignore

        @validator("box")
        def validate_box(cls, value):
            if value is None:
                return value
            if value.shape == (3, 3):
                return value
            elif value.shape == (3,):
                value = value * np.eye(3)
                return value
            else:
                raise InvalidBoxError

        @validator("topology")
        def validate_topology(cls, value):
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
                    f"Found object of type {type(value)}."
                )

    def __init__(self):
        self._inner_data = self._InnerSystem()

    @property
    def handlers(self):
        """Get the PotentialHandler objects in this Interchange object."""
        return self._inner_data.handlers

    def add_handler(self, handler_name: str, handler):
        """Add a ParameterHandler to this Interchange object."""
        self._inner_data.handlers.update({handler_name: handler})

    def remove_handler(self, handler_name: str):
        """Remove a PotentialHandler in this Interchange object."""
        self._inner_data.handlers.pop(handler_name)

    @property
    def topology(self):
        """Get the OpenFF Topology object in this Interchange object."""
        return self._inner_data.topology

    @topology.setter
    def topology(self, value):
        self._inner_data.topology = value

    @property
    def positions(self):
        """Get the positions of all particles."""
        return self._inner_data.positions

    @positions.setter
    def positions(self, value):
        self._inner_data.positions = value

    @property
    def velocities(self):
        """Get the velocities of all particles."""
        return self._inner_data.velocities

    @velocities.setter
    def velocities(self, value):
        self._inner_data.velocities = value

    @property
    def box(self):
        """If periodic, an array representing the periodic boundary conditions."""
        return self._inner_data.box

    @box.setter
    def box(self, value):
        self._inner_data.box = value

    @classmethod
    def _check_supported_handlers(cls, force_field: ForceField):

        unsupported = list()

        for handler_name in force_field.registered_parameter_handlers:
            if handler_name in {"ToolkitAM1BCC"}:
                continue
            if handler_name not in _SUPPORTED_SMIRNOFF_HANDLERS:
                unsupported.append(handler_name)

        if unsupported:
            raise SMIRNOFFHandlersNotImplementedError(unsupported)

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
            [molecule.conformers[0] for molecule in self.topology.molecules]
        )

    @classmethod
    def from_smirnoff(
        cls,
        force_field: ForceField,
        topology: Union[Topology, List[Molecule]],
        box=None,
        charge_from_molecules: Optional[List[Molecule]] = None,
        partial_bond_orders_from_molecules: Optional[List[Molecule]] = None,
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
        charge_from_molecules : `List[openff.toolkit.molecule.Molecule]`, optional
            If specified, partial charges will be taken from the given molecules
            instead of being determined by the force field.
        partial_bond_orders_from_molecules : List[openff.toolkit.molecule.Molecule], optional
            If specified, partial bond orders will be taken from the given molecules
            instead of being determined by the force field.

        Notes
        -----
        If the `Molecule` objects in the `topology` argument each contain conformers, the returned `Interchange` object
        will have its positions set via concatenating the 0th conformer of each `Molecule`.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        OpenFF 1.0.0 "Parsley"

        .. code-block:: pycon

            >>> from openff.interchange import Interchange
            >>> from openff.toolkit.topology import Molecule, Topology
            >>> from openff.toolkit.typing.engines.smirnoff import ForceField
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> parsley = ForceField("openff-1.0.0.offxml")
            >>> interchange = Interchange.from_smirnoff(topology=[mol], force_field=parsley)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        from openff.interchange.components.smirnoff import (
            SMIRNOFF_POTENTIAL_HANDLERS,
            SMIRNOFFBondHandler,
            SMIRNOFFConstraintHandler,
            SMIRNOFFElectrostaticsHandler,
            SMIRNOFFProperTorsionHandler,
            SMIRNOFFVirtualSiteHandler,
        )

        sys_out = Interchange()

        sys_out.topology = topology

        sys_out.positions = sys_out._infer_positions()

        cls._check_supported_handlers(force_field)

        if "Electrostatics" not in force_field.registered_parameter_handlers:
            if _check_electrostatics_handlers(force_field):
                raise MissingParameterHandlerError(
                    "Force field contains parameter handler(s) that may assign/modify "
                    "partial charges, but no ElectrostaticsHandler was found."
                )

        parameter_handlers_by_type = {
            force_field[parameter_handler_name].__class__: force_field[
                parameter_handler_name
            ]
            for parameter_handler_name in force_field.registered_parameter_handlers
        }

        if len(parameter_handlers_by_type) != len(
            force_field.registered_parameter_handlers
        ):

            raise NotImplementedError(
                "Only force fields that contain one instance of each parameter handler "
                "type are currently supported."
            )

        for potential_handler_type in SMIRNOFF_POTENTIAL_HANDLERS:

            parameter_handlers = [
                parameter_handlers_by_type[allowed_type]
                for allowed_type in potential_handler_type.allowed_parameter_handlers()
                if allowed_type in parameter_handlers_by_type
            ]

            if len(parameter_handlers) == 0:
                continue

            # TODO: Might be simpler to rework the bond handler to be self-contained and
            #       move back to the constraint handler dealing with the logic (and
            #       depending on the bond handler)
            if potential_handler_type == SMIRNOFFBondHandler:
                SMIRNOFFBondHandler.check_supported_parameters(force_field["Bonds"])
                bond_handler = SMIRNOFFBondHandler._from_toolkit(
                    parameter_handler=force_field["Bonds"],
                    topology=sys_out._inner_data.topology,
                    # constraint_handler=constraint_handler,
                    partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
                )
                sys_out.handlers.update({"Bonds": bond_handler})
            elif potential_handler_type == SMIRNOFFProperTorsionHandler:
                SMIRNOFFProperTorsionHandler.check_supported_parameters(
                    force_field["ProperTorsions"]
                )
                potential_handler = SMIRNOFFProperTorsionHandler._from_toolkit(
                    parameter_handler=force_field["ProperTorsions"],
                    topology=sys_out._inner_data.topology,
                    partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
                )
                sys_out.handlers.update({"ProperTorsions": potential_handler})
            elif potential_handler_type == SMIRNOFFConstraintHandler:
                (bond_handler, constraint_handler) = (
                    force_field._parameter_handlers.get(val, None)
                    for val in ["Bonds", "Constraints"]
                )
                if constraint_handler is None:
                    continue
                constraints = SMIRNOFFConstraintHandler._from_toolkit(
                    parameter_handler=[
                        val
                        for val in [bond_handler, constraint_handler]
                        if val is not None
                    ],
                    topology=sys_out._inner_data.topology,
                )
                sys_out.handlers.update({"Constraints": constraints})
            elif potential_handler_type == SMIRNOFFElectrostaticsHandler:
                electrostatics_handler = SMIRNOFFElectrostaticsHandler._from_toolkit(
                    parameter_handler=parameter_handlers,
                    topology=sys_out._inner_data.topology,
                    charge_from_molecules=charge_from_molecules,
                )
                sys_out.handlers.update({"Electrostatics": electrostatics_handler})
            elif potential_handler_type == SMIRNOFFVirtualSiteHandler:
                virtual_site_handler = SMIRNOFFVirtualSiteHandler._from_toolkit(
                    parameter_handler=force_field["VirtualSites"],
                    topology=sys_out._inner_data.topology,
                )
                virtual_site_handler.exclusion_policy = force_field[
                    "VirtualSites"
                ].exclusion_policy
                sys_out.handlers.update({"VirtualSites": virtual_site_handler})
                sys_out["vdW"]._from_toolkit_virtual_sites(
                    parameter_handler=force_field["VirtualSites"],
                    topology=sys_out._inner_data.topology,
                )
                sys_out["Electrostatics"]._from_toolkit_virtual_sites(
                    parameter_handler=force_field["VirtualSites"],
                    topology=sys_out._inner_data.topology,
                )
            elif len(potential_handler_type.allowed_parameter_handlers()) > 1:
                potential_handler = potential_handler_type._from_toolkit(  # type: ignore
                    parameter_handler=parameter_handlers,
                    topology=sys_out._inner_data.topology,
                )
                sys_out.handlers.update({potential_handler.type: potential_handler})
            else:
                potential_handler_type.check_supported_parameters(parameter_handlers[0])
                potential_handler = potential_handler_type._from_toolkit(  # type: ignore
                    parameter_handler=parameter_handlers[0],
                    topology=sys_out._inner_data.topology,
                )
                sys_out.handlers.update({potential_handler.type: potential_handler})

        # `box` argument is only overriden if passed `None` and the input topology
        # is a `Topology` (could be `List[Molecule]`) and has box vectors
        if box is None:
            if isinstance(topology, Topology):
                sys_out.box = topology.box_vectors
            else:
                sys_out.box = None
        else:
            sys_out.box = box

        return sys_out

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
                "Cannot visualize system without positions."
            ) from error
        return nglview.show_file("_tmp_pdb_file.pdb")

    def to_gro(self, file_path: Union[Path, str], writer="internal", decimal: int = 8):
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

    def to_openmm(self, combine_nonbonded_forces: bool = False):
        """Export this Interchange to an OpenMM System."""
        from openff.interchange.interop.openmm import to_openmm as to_openmm_

        return to_openmm_(self, combine_nonbonded_forces=combine_nonbonded_forces)

    def to_openmm_topology(self):
        """Export components of this Interchange to an OpenMM Topology."""
        from openff.interchange.interop.openmm import to_openmm_topology

        return to_openmm_topology(self)

    def to_prmtop(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to an Amber .prmtop file."""
        if writer == "internal":
            from openff.interchange.interop.internal.amber import to_prmtop

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
                "Positions are required to write a `.pdb` file but found None."
            )

        if writer == "openmm":
            from openff.interchange.interop.openmm import _to_pdb

            _to_pdb(file_path, self.topology, self.positions)
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
            from openff.interchange.interop.internal.amber import to_inpcrd

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
        cls, force_field: "FoyerForcefield", topology: "Topology", **kwargs
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

            system.handlers[name] = handler

        system.handlers["vdW"].store_matches(force_field, topology=system.topology)
        system.handlers["vdW"].store_potentials(force_field=force_field)

        atom_slots = system.handlers["vdW"].slot_map

        system.handlers["Electrostatics"].store_charges(
            atom_slots=atom_slots,
            force_field=force_field,
        )

        system.handlers["vdW"].scale_14 = force_field.lj14scale
        system.handlers["Electrostatics"].scale_14 = force_field.coulomb14scale

        for name, handler in system.handlers.items():
            if name not in ["vdW", "Electrostatics"]:
                handler.store_matches(atom_slots, topology=system.topology)
                handler.store_potentials(force_field)

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
            for key in via_intermol.handlers:
                if key not in [
                    "Bonds",
                    "Angles",
                    "ProperTorsions",
                    "ImproperTorsions",
                    "vdW",
                    "Electrostatics",
                ]:
                    raise Exception(f"Found unexpected handler with name {key}")
                    via_internal.handlers[key] = via_intermol.handlers[key]

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
        for handler in self.handlers:
            if handler == handler_name:
                return self[handler_name]._get_parameters(atom_indices=atom_indices)
        raise MissingParameterHandlerError(
            f"Could not find parameter handler of name {handler_name}"
        )

    def _get_nonbonded_methods(self):
        if "vdW" in self.handlers:
            nonbonded_handler = "vdW"
        elif "Buckingham-6" in self.handlers:
            nonbonded_handler = "Buckingham-6"
        else:
            raise InternalInconsistencyError("Found no non-bonded handlers")

        nonbonded_ = {
            "electrostatics_method": self.handlers["Electrostatics"].method,
            "vdw_method": self.handlers[nonbonded_handler].method,
            "periodic_topology": self.box is not None,
        }

        return nonbonded_

    # TODO: Does this cause any strange behaviors with Pydantic?
    # Taken from https://stackoverflow.com/a/4017638/4248961
    _aliases = {"box_vectors": "x", "coordinates": "positions", "top": "topology"}

    def __setattr__(self, name, value):
        name = self._aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        name = self._aliases.get(name, name)
        return object.__getattribute__(self, name)

    def __getitem__(self, item: str):
        """Syntax sugar for looking up potential handlers or other components."""
        if type(item) != str:
            raise LookupError(
                "Only str arguments can be currently be used for lookups.\n"
                f"Found item {item} of type {type(item)}"
            )
        if item == "positions":
            return self.positions
        elif item in {"box", "box_vectors"}:
            return self.box
        elif item in self.handlers:
            return self.handlers[item]
        else:
            raise LookupError(
                f"Could not find component {item}. This object has the following "
                f"potential handlers registered:\n\t{[*self.handlers.keys()]}"
            )

    def __add__(self, other):
        """Combine two Interchange objects. This method is unstable and likely unsafe."""
        from openff.interchange.components.toolkit import _combine_topologies
        from openff.interchange.models import TopologyKey

        warnings.warn(
            "Interchange object combination is experimental and likely to produce "
            "strange results. Any workflow using this method is not guaranteed to "
            "be suitable for production. Use with extreme caution and thoroughly "
            "validate results!"
        )

        self_copy = Interchange()
        self_copy._inner_data = deepcopy(self._inner_data)

        self_copy.topology = _combine_topologies(self.topology, other.topology)
        atom_offset = self.topology.n_atoms

        """
        for handler_name in self.handlers:
            if type(self.handlers[handler_name]).__name__ == "FoyerElectrostaticsHandler":
                self.handlers[handler_name].slot_map = self.handlers[handler_name].charges
        """

        for handler_name, handler in other.handlers.items():

            """
            if type(handler).__name__ == "FoyerElectrostaticsHandler":
                handler.slot_map = handler.charges
            """

            # TODO: Actually specify behavior in this case
            try:
                self_handler = self_copy.handlers[handler_name]
            except KeyError:
                self.add_handler(handler_name, handler)
                warnings.warn(
                    f"'other' Interchange object has handler with name {handler_name} not "
                    f"found in 'self,' but it has now been added."
                )
                continue

            for top_key, pot_key in handler.slot_map.items():

                new_atom_indices = tuple(
                    idx + atom_offset for idx in top_key.atom_indices
                )
                new_top_key = TopologyKey(
                    atom_indices=new_atom_indices,
                    mult=top_key.mult,
                )

                self_handler.slot_map.update({new_top_key: pot_key})
                if handler_name == "Constraints":
                    self_handler.constraints.update(
                        {pot_key: handler.constraints[pot_key]}
                    )
                else:
                    self_handler.potentials.update(
                        {pot_key: handler.potentials[pot_key]}
                    )

        if self_copy.positions is not None and other.positions is not None:
            new_positions = np.vstack([self_copy.positions, other.positions])
            self_copy.positions = new_positions
        else:
            warnings.warn(
                "Setting positions to None because one or both objects added together were missing positions."
            )
            self_copy.positions = None

        if not np.all(self_copy.box == other.box):
            raise UnsupportedCombinationError(
                "Combination with unequal box vectors is not curretnly supported"
            )

        return self_copy

    def __repr__(self):
        periodic = self.box is not None
        n_atoms = self.topology.n_atoms
        return f"Interchange with {n_atoms} atoms, {'' if periodic else 'non-'}periodic topology"
