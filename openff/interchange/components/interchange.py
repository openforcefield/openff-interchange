"""An object for storing, manipulating, and converting molecular mechanics data."""

import copy
import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, overload

import numpy as np
from openff.models.models import DefaultModel
from openff.models.types import ArrayQuantity, QuantityEncoder
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.utilities.utilities import has_package, requires_package

from openff.interchange._experimental import experimental
from openff.interchange._pydantic import Field, validator
from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ImproperTorsionCollection,
    ProperTorsionCollection,
)
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.components.potentials import Collection
from openff.interchange.exceptions import (
    InvalidBoxError,
    InvalidTopologyError,
    MissingParameterHandlerError,
    MissingPositionsError,
    UnsupportedExportError,
)
from openff.interchange.operations.minimize import (
    _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
)
from openff.interchange.smirnoff import (
    SMIRNOFFConstraintCollection,
    SMIRNOFFVirtualSiteCollection,
)
from openff.interchange.warnings import InterchangeDeprecationWarning

if has_package("foyer"):
    from foyer.forcefield import Forcefield as FoyerForcefield
if has_package("nglview"):
    import nglview

if TYPE_CHECKING:
    import openmm
    import openmm.app


class TopologyEncoder(json.JSONEncoder):
    """Custom encoder for `Topology` objects."""

    def default(self, obj: Topology):
        """Encode a `Topology` object to JSON."""
        _topology = copy.deepcopy(obj)
        for molecule in _topology.molecules:
            molecule._conformers = None

        return _topology.to_json()


def interchange_dumps(v, *, default):
    """Dump an Interchange to JSON after converting to compatible types."""
    from openff.interchange.smirnoff._base import dump_collection

    return json.dumps(
        {
            "positions": QuantityEncoder().default(v["positions"]),
            "box": QuantityEncoder().default(v["box"]),
            "topology": TopologyEncoder().default(v["topology"]),
            "collections": {
                key: dump_collection(v["collections"][key], default=default)
                for key in v["collections"]
            },
        },
        default=default,
    )


def interchange_loader(data: str) -> dict:
    """Load a JSON representation of an Interchange object."""
    tmp: dict[str, int | bool | str | dict | None] = {}

    for key, val in json.loads(data).items():
        if val is None:
            continue
        if key == "positions":
            tmp["positions"] = Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "velocities":
            tmp["velocities"] = Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "box":
            tmp["box"] = Quantity(val["val"], unit.Unit(val["unit"]))
        elif key == "topology":
            tmp["topology"] = Topology.from_json(val)
        elif key == "collections":
            from openff.interchange.smirnoff import (
                SMIRNOFFAngleCollection,
                SMIRNOFFBondCollection,
                SMIRNOFFConstraintCollection,
                SMIRNOFFElectrostaticsCollection,
                SMIRNOFFImproperTorsionCollection,
                SMIRNOFFProperTorsionCollection,
                SMIRNOFFvdWCollection,
                SMIRNOFFVirtualSiteCollection,
            )

            tmp["collections"] = {}

            _class_mapping = {
                "Bonds": SMIRNOFFBondCollection,
                "Angles": SMIRNOFFAngleCollection,
                "Constraints": SMIRNOFFConstraintCollection,
                "ProperTorsions": SMIRNOFFProperTorsionCollection,
                "ImproperTorsions": SMIRNOFFImproperTorsionCollection,
                "vdW": SMIRNOFFvdWCollection,
                "Electrostatics": SMIRNOFFElectrostaticsCollection,
                "VirtualSites": SMIRNOFFVirtualSiteCollection,
            }

            for collection_name, collection_data in val.items():
                tmp["collections"][collection_name] = _class_mapping[  # type: ignore
                    collection_name
                ].parse_raw(collection_data)

    return tmp


class Interchange(DefaultModel):
    """
    A object for storing, manipulating, and converting molecular mechanics data.

    .. warning :: This object is in an early and experimental state and unsuitable for production.
    .. warning :: This API is experimental and subject to change.
    """

    collections: dict[str, Collection] = Field(dict())
    topology: Topology = Field(None)
    mdconfig: MDConfig = Field(None)
    box: ArrayQuantity["nanometer"] = Field(None)
    positions: ArrayQuantity["nanometer"] = Field(None)
    velocities: ArrayQuantity["nanometer / picosecond"] = Field(None)

    class Config:
        """Custom Pydantic-facing configuration for the Interchange class."""

        json_loads = interchange_loader
        json_dumps = interchange_dumps
        validate_assignment = True
        arbitrary_types_allowed = True

    @validator("box", allow_reuse=True)
    def validate_box(cls, value) -> Quantity | None:
        if value is None:
            return value

        validated = ArrayQuantity.validate_type(value)

        dimensions = np.atleast_2d(validated).shape

        if dimensions == (3, 3):
            return validated
        elif dimensions == (1, 3):
            return validated * np.eye(3)
        else:
            raise InvalidBoxError(
                f"Failed to convert value {value} to 3x3 box vectors. Please file an issue if you think this "
                "input should be supported and the failure is an error.",
            )

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
                "Could not process topology argument, expected openff.toolkit.Topology. "
                f"Found object of type {type(value)}.",
            )

    def _infer_positions(self) -> Quantity | None:
        """
        Attempt to set Interchange.positions based on conformers in molecules in the topology.

        If _any_ molecule lacks conformers, return None.
        If _all_ molecules have conformers, return an array of shape (self.topology.n_atoms, 3)
        generated by concatenating the positions of each molecule, using only the 0th conformer.
        """
        from openff.interchange.common._positions import _infer_positions

        return _infer_positions(self.topology, self.positions)

    @classmethod
    def from_smirnoff(
        cls,
        force_field: ForceField,
        topology: Topology | list[Molecule],
        box=None,
        positions=None,
        charge_from_molecules: list[Molecule] | None = None,
        partial_bond_orders_from_molecules: list[Molecule] | None = None,
        allow_nonintegral_charges: bool = False,
    ) -> "Interchange":
        """
        Create a new object by parameterizing a topology with a SMIRNOFF force field.

        Parameters
        ----------
        force_field : `openff.toolkit.ForceField`
            The force field to parameterize the topology with.
        topology : `openff.toolkit.Topology` or `List[openff.toolkit.Molecule]`
            The topology to parameterize, or a list of molecules to construct a
            topology from and parameterize.
        box : `openff.units.Quantity`, optional
            The box vectors associated with the ``Interchange``. If ``None``,
            box vectors are taken from the topology, if present.
        positions : `openff.units.Quantity`, optional
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
            >>> from openff.toolkit import ForceField, Molecule
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

    def visualize(
        self,
        backend: str = "nglview",
        include_virtual_sites: bool = False,
    ) -> "nglview.NGLWidget":
        """
        Visualize this Interchange.

        This currently only uses NGLview. Other engines may be added in the future.

        Parameters
        ----------
        backend : str, default="nglview"
            The backend to use for visualization. Currently only "nglview" is supported.
        include_virtual_sites : bool, default=False
            Whether or not to include virtual sites in the visualization.

        Returns
        -------
        widget : nglview.NGLWidget
            The NGLWidget containing the visualization.

        """
        from openff.toolkit.utils.exceptions import (
            IncompatibleUnitError,
            MissingConformersError,
        )

        if backend == "nglview":
            if include_virtual_sites:

                return self._visualize_nglview(include_virtual_sites=True)

            else:

                # Interchange.topology might have its own positions;
                # just use Interchange.positions
                original_positions = self.topology.get_positions()

                try:
                    self.topology.set_positions(self.positions)
                    widget = self.topology.visualize()
                except (
                    MissingConformersError,
                    IncompatibleUnitError,
                    ValueError,
                ) as error:
                    raise MissingPositionsError(
                        "Cannot visualize system without positions.",
                    ) from error

                # but don't modify them long-term
                # work around https://github.com/openforcefield/openff-toolkit/issues/1820
                if original_positions is not None:
                    self.topology.set_positions(original_positions)
                else:
                    for molecule in self.topology.molecules:
                        molecule._conformers = None

                return widget

        else:

            raise UnsupportedExportError

    @requires_package("nglview")
    def _visualize_nglview(
        self,
        include_virtual_sites: bool = False,
    ) -> "nglview.NGLWidget":
        """
        Visualize the system using NGLView via a PDB file.

        include_virtual_sites : bool, default=False
            Whether or not to include virtual sites in the visualization.
        """
        import nglview

        from openff.interchange.components._viz import InterchangeNGLViewStructure

        try:
            widget = nglview.NGLWidget(
                InterchangeNGLViewStructure(
                    interchange=self,
                    ext="pdb",
                ),
                representations=[
                    dict(type="unitcell", params=dict()),
                ],
            )

        except MissingPositionsError as error:
            raise MissingPositionsError(
                "Cannot visualize system without positions.",
            ) from error

        widget.add_representation("line", sele="water")
        widget.add_representation("spacefill", sele="ion")
        widget.add_representation("cartoon", sele="protein")
        widget.add_representation(
            "licorice",
            sele="not water and not ion and not protein",
            radius=0.25,
            multipleBond=False,
        )

        return widget

    def minimize(
        self,
        engine: str = "openmm",
        force_tolerance: Quantity = _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
        max_iterations: int = 10_000,
    ):
        """
        Minimize the energy of the system using an available engine.

        Updates positions in-place.

        Parameters
        ----------
        engine : str, default="openmm"
            The engine to use for minimization. Currently only "openmm" is supported.
        force_tolerance : openff.units.Quantity, default=10.0 kJ / mol / nm
            The force tolerance to run until during energy minimization.
        max_iterations : int, default=10_000
            The maximum number of iterations to run during energy minimization.

        """
        if engine == "openmm":
            from openff.interchange.operations.minimize.openmm import minimize_openmm

            minimized_positions = minimize_openmm(
                self,
                tolerance=force_tolerance,
                max_iterations=max_iterations,
            )
            self.positions = minimized_positions
        else:
            raise NotImplementedError(f"Engine {engine} is not implemented.")

    def to_gromacs(
        self,
        prefix: str,
        decimal: int = 3,
        hydrogen_mass: float = 1.007947,
        merge_atom_types: bool = False,
    ):
        """
        Export this Interchange object to GROMACS files.

        Parameters
        ----------
        prefix: str
            The prefix to use for the GROMACS topology and coordinate files, i.e. "foo" will produce
            "foo.top" and "foo.gro".
        decimal: int, default=3
            The number of decimal places to use when writing the GROMACS coordinate file.
        hydrogen_mass : float, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters and is unsupported when virtual sites are present.
        merge_atom_types: bool, default = False
            The flag to define behaviour of GROMACSWriter. If True, then similar atom types will be merged.
            If False, each atom will have its own atom type.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        writer = GROMACSWriter(
            system=_convert(self, hydrogen_mass=hydrogen_mass),
            top_file=prefix + ".top",
            gro_file=prefix + ".gro",
        )

        writer.to_top(merge_atom_types=merge_atom_types)
        writer.to_gro(decimal=decimal)

    def to_top(
        self,
        file_path: Path | str,
        hydrogen_mass: float = 1.007947,
    ):
        """
        Export this Interchange to a GROMACS topology file.

        Parameters
        ----------
        file_path
            The path to the GROMACS topology file to write.
        hydrogen_mass : float, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters and is unsupported when virtual sites are present.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        GROMACSWriter(
            system=_convert(self, hydrogen_mass=hydrogen_mass),
            top_file=file_path,
        ).to_top()

    def to_gro(self, file_path: Path | str, decimal: int = 3):
        """
        Export this Interchange object to a GROMACS coordinate file.

        Parameters
        ----------
        file_path: Union[Path, str]
            The path to the GROMACS coordinate file to write.
        decimal: int, default=3
            The number of decimal places to use when writing the GROMACS coordinate file.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        GROMACSWriter(
            system=_convert(self),
            gro_file=file_path,
        ).to_gro(decimal=decimal)

    def to_lammps(self, file_path: Path | str, writer="internal"):
        """Export this Interchange to a LAMMPS data file."""
        if writer == "internal":
            from openff.interchange.interop.lammps import to_lammps

            to_lammps(self, file_path)
        else:
            raise UnsupportedExportError

    def to_openmm_system(
        self,
        combine_nonbonded_forces: bool = True,
        add_constrained_forces: bool = False,
        ewald_tolerance: float = 1e-4,
        hydrogen_mass: float = 1.007947,
    ):
        """
        Export this Interchange to an OpenMM System.

        Parameters
        ----------
        combine_nonbonded_forces : bool, default=True
            If True, an attempt will be made to combine all non-bonded interactions into a single
            openmm.NonbondedForce.
            If False, non-bonded interactions will be split across multiple forces.
        add_constrained_forces : bool, default=False,
            If True, add valence forces that might be overridden by constraints, i.e. call `addBond` or `addAngle`
            on a bond or angle that is fully constrained.
        ewald_tolerance : float, default=1e-4
            The value passed to `NonbondedForce.setEwaldErrorTolerance`
        hydrogen_mass : float, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters and is unsupported when virtual sites are present.

        Returns
        -------
        system : openmm.System
            The OpenMM System object.

        """
        from openff.interchange.interop.openmm import (
            to_openmm_system as _to_openmm_system,
        )

        return _to_openmm_system(
            self,
            combine_nonbonded_forces=combine_nonbonded_forces,
            add_constrained_forces=add_constrained_forces,
            ewald_tolerance=ewald_tolerance,
            hydrogen_mass=hydrogen_mass,
        )

    to_openmm = to_openmm_system

    def to_openmm_topology(
        self,
        ensure_unique_atom_names: str | bool = "residues",
    ):
        """Export components of this Interchange to an OpenMM Topology."""
        from openff.interchange.interop.openmm._topology import to_openmm_topology

        return to_openmm_topology(
            self,
            ensure_unique_atom_names=ensure_unique_atom_names,
        )

    def to_openmm_simulation(
        self,
        integrator: "openmm.Integrator",
        combine_nonbonded_forces: bool = True,
        add_constrained_forces: bool = False,
        additional_forces: Iterable["openmm.Force"] = tuple(),
        **kwargs,
    ) -> "openmm.app.simulation.Simulation":
        """
        Export this Interchange to an OpenMM `Simulation` object.

        Positions are set on the `Simulation` if present on the `Interchange`.

        Parameters
        ----------
        integrator : subclass of openmm.Integrator
            The integrator to use for the simulation.
        combine_nonbonded_forces : bool, default=False
            If True, an attempt will be made to combine all non-bonded interactions into a single
            openmm.NonbondedForce.
            If False, non-bonded interactions will be split across multiple forces.
        add_constrained_forces : bool, default=False,
            If True, add valence forces that might be overridden by constraints, i.e. call `addBond` or `addAngle`
            on a bond or angle that is fully constrained.
        additional_forces : Iterable[openmm.Force], default=tuple()
            Additional forces to be added to the system, i.e. barostats that are not
            added by the force field.
        **kwargs
            Further keyword parameters are passed on to
            :py:meth:`Simulation.__init__() <openmm.app.simulation.Simulation.__init__>`

        Returns
        -------
        simulation : openmm.app.Simulation
            The OpenMM simulation object, possibly with positions set.

        Examples
        --------
        Create an OpenMM simulation with a Langevin integrator:

        >>> import openmm
        >>> import openmm.unit
        >>>
        >>> integrator = openmm.LangevinMiddleIntegrator(
        ...     293.15 * openmm.unit.kelvin,
        ...     1.0 / openmm.unit.picosecond,
        ...     2.0 * openmm.unit.femtosecond,
        ... )
        >>> barostat = openmm.MonteCarloBarostat(
        ...     1.00 * openmm.unit.bar,
        ...     293.15 * openmm.unit.kelvin,
        ...     25,
        ... )
        >>> simulation = interchange.to_openmm_simulation(
        ...     integrator=integrator,
        ...     additional_forces=[barostat],
        ... )

        Add a barostat:


        Re-initializing the `Context` after adding a `Force` is necessary due to implementation details in OpenMM.
        For more, see
        https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#why-does-it-ignore-changes-i-make-to-a-system-or-force

        """
        import openmm.app

        from openff.interchange.interop.openmm._positions import to_openmm_positions

        system = self.to_openmm_system(
            combine_nonbonded_forces=combine_nonbonded_forces,
            add_constrained_forces=add_constrained_forces,
        )

        for force in additional_forces:
            system.addForce(force)

        # since we're adding forces before making a context, we don't need to
        # re-initialize context. In a different order, we would need to:
        # https://github.com/openforcefield/openff-interchange/pull/725#discussion_r1210928501

        simulation = openmm.app.Simulation(
            topology=self.to_openmm_topology(),
            system=system,
            integrator=integrator,
            **kwargs,
        )

        # If the system contains virtual sites, the positions must, so no obvious case in which
        # include_virtual_sites could possibly be False
        if self.positions is not None:
            simulation.context.setPositions(
                to_openmm_positions(self, include_virtual_sites=True),
            )

        return simulation

    def to_prmtop(self, file_path: Path | str, writer="internal"):
        """Export this Interchange to an Amber .prmtop file."""
        if writer == "internal":
            from openff.interchange.interop.amber import to_prmtop

            to_prmtop(self, file_path)

        else:
            raise UnsupportedExportError

    @requires_package("openmm")
    def to_pdb(self, file_path: Path | str, include_virtual_sites: bool = False):
        """Export this Interchange to a .pdb file."""
        from openff.interchange.interop.openmm import _to_pdb

        if self.positions is None:
            raise MissingPositionsError(
                "Positions are required to write a `.pdb` file but found None.",
            )

        # TODO: Simply wire `include_virtual_sites` to `to_openmm_{topology|positions}`?
        if include_virtual_sites:
            from openff.interchange.interop._virtual_sites import (
                get_positions_with_virtual_sites,
            )

            openmm_topology = self.to_openmm_topology(
                ensure_unique_atom_names=False,
            )
            positions = get_positions_with_virtual_sites(self)

        else:
            openmm_topology = self.topology.to_openmm(
                ensure_unique_atom_names=False,
            )
            positions = self.positions

        _to_pdb(file_path, openmm_topology, positions.to(unit.angstrom))

    def to_psf(self, file_path: Path | str):
        """Export this Interchange to a CHARMM-style .psf file."""
        raise UnsupportedExportError

    def to_crd(self, file_path: Path | str):
        """Export this Interchange to a CHARMM-style .crd file."""
        raise UnsupportedExportError

    def to_inpcrd(self, file_path: Path | str, writer="internal"):
        """Export this Interchange to an Amber .inpcrd file."""
        if writer == "internal":
            from openff.interchange.interop.amber import to_inpcrd

            to_inpcrd(self, file_path)

        else:
            raise UnsupportedExportError

    @classmethod
    @requires_package("foyer")
    def from_foyer(
        cls,
        force_field: "FoyerForcefield",
        topology: Topology,
        box=None,
        positions=None,
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
            >>> from openff.toolkit import Molecule, Topology
            >>> from foyer import Forcefield
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> top = Topology.from_molecules([mol])
            >>> oplsaa = Forcefield(name="oplsaa")
            >>> interchange = Interchange.from_foyer(topology=top, force_field=oplsaa)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        from openff.interchange.foyer._create import _create_interchange

        return _create_interchange(
            force_field=force_field,
            topology=topology,
            box=box,
            positions=positions,
        )

    @classmethod
    @experimental
    def from_gromacs(
        cls,
        topology_file: Path | str,
        gro_file: Path | str,
    ) -> "Interchange":
        """
        Create an Interchange object from GROMACS files.

        WARNING! This method is experimental and not suitable for production.

        Parameters
        ----------
        topology_file : Union[Path, str]
            The path to a GROMACS topology file.
        gro_file : Union[Path, str]
            The path to a GROMACS coordinate file.

        Returns
        -------
        interchange : Interchange
            An Interchange object representing the contents of the GROMACS files.

        """
        from openff.interchange.interop.gromacs._import._import import from_files
        from openff.interchange.interop.gromacs._interchange import to_interchange

        return to_interchange(
            from_files(
                top_file=topology_file,
                gro_file=gro_file,
            ),
        )

    @classmethod
    @experimental
    def from_openmm(
        cls,
        system: "openmm.System",
        topology: Union["openmm.app.Topology", Topology, None] = None,
        positions: Quantity | None = None,
        box_vectors: Quantity | None = None,
    ) -> "Interchange":
        """
        Create an Interchange object from OpenMM objects.

        WARNING! This method is experimental and not suitable for production.

        Parameters
        ----------
        system : openmm.System, optional
            The OpenMM system.
        topology : openmm.app.Topology, optional
            The OpenMM topology.
        positions : openmm.unit.Quantity or openff.units.Quantity, optional
            The positions of particles in this system and/or topology.
        box_vectors : openmm.unit.Quantity or openff.units.Quantity, optional
            The vectors of the simulation box associated with this system and/or topology.

        Returns
        -------
        interchange : Interchange
            An Interchange object representing the contents of the OpenMM objects.

        """
        from openff.interchange.interop.openmm._import._import import from_openmm

        return from_openmm(
            topology=topology,
            system=system,
            positions=positions,
            box_vectors=box_vectors,
        )

    def _get_parameters(self, handler_name: str, atom_indices: tuple[int]) -> dict:
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
                stacklevel=2,
            )
            return self.collections
        else:
            return super().__getattribute__(attr)

    @overload
    def __getitem__(self, item: Literal["Bonds"]) -> "BondCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["Constraints"],
    ) -> "SMIRNOFFConstraintCollection": ...

    @overload
    def __getitem__(self, item: Literal["Angles"]) -> "AngleCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["vdW"],
    ) -> "vdWCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["ProperTorsions"],
    ) -> "ProperTorsionCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["ImproperTorsions"],
    ) -> "ImproperTorsionCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["VirtualSites"],
    ) -> "SMIRNOFFVirtualSiteCollection": ...

    @overload
    def __getitem__(
        self,
        item: Literal["Electrostatics"],
    ) -> "ElectrostaticsCollection": ...

    @overload
    def __getitem__(self, item: str) -> "Collection": ...

    def __getitem__(self, item: str):
        """Syntax sugar for looking up collections or other components."""
        if type(item) is not str:
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

    @experimental
    def __add__(self, other: "Interchange") -> "Interchange":
        """Combine two Interchange objects. This method is unstable and not yet unsafe."""
        warnings.warn(
            "The `+` operator is deprecated. Use `Interchange.combine` instead.",
            InterchangeDeprecationWarning,
            stacklevel=2,
        )

        return self.combine(other)

    @experimental
    def combine(self, other: "Interchange") -> "Interchange":
        """Combine two Interchange objects. This method is unstable and not yet unsafe."""
        from openff.interchange.operations._combine import _combine

        return _combine(self, other)

    def __repr__(self) -> str:
        periodic = self.box is not None
        n_atoms = self.topology.n_atoms
        return (
            f"Interchange with {len(self.collections)} collections, "
            f"{'' if periodic else 'non-'}periodic topology with {n_atoms} atoms."
        )
