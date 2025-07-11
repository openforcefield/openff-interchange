"""An object for storing, manipulating, and converting molecular mechanics data."""

import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, overload

from openff.toolkit import Molecule, Quantity, Topology, unit
from openff.utilities.utilities import has_package, requires_package
from pydantic import Field

from openff.interchange._annotations import (
    PositiveFloat,
    _BoxQuantity,
    _PositionsQuantity,
    _VelocityQuantity,
)
from openff.interchange._experimental import experimental
from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ImproperTorsionCollection,
    ProperTorsionCollection,
)
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.components.potentials import Collection, _AnnotatedCollections
from openff.interchange.exceptions import (
    InvalidPositionsError,
    MissingParameterHandlerError,
    MissingPositionsError,
    UnsupportedExportError,
)
from openff.interchange.operations.minimize import (
    _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
)
from openff.interchange.pydantic import _BaseModel
from openff.interchange.serialization import _AnnotatedTopology
from openff.interchange.smirnoff import (
    SMIRNOFFConstraintCollection,
    SMIRNOFFVirtualSiteCollection,
)
from openff.interchange.smirnoff._gbsa import SMIRNOFFGBSACollection
from openff.interchange.warnings import InterchangeDeprecationWarning

if TYPE_CHECKING:
    import openmm
    import openmm.app
    from openff.toolkit import ForceField

    from openff.interchange.foyer._guard import has_foyer

    if has_foyer:
        try:
            from foyer import Forcefield as FoyerForcefield
        except ModuleNotFoundError:
            # case of openff/interchange/foyer/ being detected as the real package
            pass
    if has_package("nglview"):
        import nglview


class Interchange(_BaseModel):
    """
    A object for storing, manipulating, and converting molecular mechanics data.

    Examples
    --------
    Create an ``Interchange`` from an OpenFF ``ForceField`` and ``Molecule``

    >>> from openff.toolkit import ForceField, Molecule
    >>> sage = ForceField("openff-2.2.0.offxml")
    >>> top = Molecule.from_smiles("CCC").to_topology()
    >>> interchange = sage.create_interchange(top)

    Get the parameters for the bond between atoms 0 and 1

    >>> interchange["Bonds"][0, 1]
    Potential(...)

    """

    collections: _AnnotatedCollections = Field(dict())
    topology: _AnnotatedTopology
    mdconfig: MDConfig | None = Field(None)
    box: _BoxQuantity | None = Field(None)  # Needs shape/OpenMM validation
    positions: _PositionsQuantity | None = Field(None)  # Ditto
    velocities: _VelocityQuantity | None = Field(None)  # Ditto

    @classmethod
    def from_smirnoff(
        cls,
        force_field: "ForceField",
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
        topology : `openff.toolkit.Topology` or `list[openff.toolkit.Molecule]`
            The topology to parameterize, or a list of molecules to construct a
            topology from and parameterize.
        box : `openff.units.Quantity`, optional
            The box vectors associated with the ``Interchange``. If ``None``,
            box vectors are taken from the topology, if present.
        positions : `openff.units.Quantity`, optional
            The positions associated with atoms in the input topology. If ``None``,
            positions are taken from the molecules in topology, if present on all molecules.
        charge_from_molecules : `list[openff.toolkit.molecule.Molecule]`, optional
            If specified, partial charges for any molecules isomorphic to those
            given will be taken from the given molecules' `partial_charges`
            attribute instead of being determined by the force field. All
            molecules in this list must have partial charges assigned and must
            not be isomorphic with any other molecules in the list. For all values
            of this argument, charges on the input topology are ignored.
        partial_bond_orders_from_molecules : list[openff.toolkit.molecule.Molecule], optional
            If specified, partial bond orders will be taken from the given molecules
            instead of being determined by the force field.
        allow_nonintegral_charges : bool, optional, default=False
            If True, allow molecules to have approximately non-integral charges.

        Notes
        -----
        If the ``Molecule`` objects in the ``topology`` argument each contain
        conformers, the returned ``Interchange`` object will have its positions
        set via concatenating the 0th conformer of each ``Molecule``.

        If the ``Molecule`` objects in the ``topology`` argument have stored
        partial charges, these are ignored and charges are assigned according to
        the contents of the force field. To override the force field and use
        preset charges, use the ``charge_from_molecules`` argument.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        OpenFF 2.0.0 "Sage"

        .. code-block:: pycon

            >>> from openff.toolkit import ForceField, Molecule
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> sage = ForceField("openff-2.0.0.offxml")
            >>> interchange = sage.create_interchange(mol.to_topology())
            >>> interchange
            Interchange with 7 collections, non-periodic topology with 8 atoms.

        """
        from openff.interchange.smirnoff._create import _create_interchange

        return _create_interchange(
            force_field=force_field,
            topology=topology,
            box=box,
            positions=positions,
            molecules_with_preset_charges=charge_from_molecules,
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

        widget.add_representation("line", selection="water")
        widget.add_representation("spacefill", selection="ion")
        widget.add_representation("cartoon", selection="protein")
        widget.add_representation(
            "licorice",
            selection="not water and not ion and not protein",
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

    def get_positions(self, include_virtual_sites: bool = True) -> Quantity:
        """
        Get the positions associated with this Interchange.

        Parameters
        ----------
        include_virtual_sites : bool, default=True
            Include virtual sites in the returned positions.

        Returns
        -------
        positions : openff.units.Quantity
            The positions of the atoms in the system.

        """
        from openff.interchange.interop.common import _to_positions

        return _to_positions(self, include_virtual_sites=include_virtual_sites)

    def to_gromacs(
        self,
        prefix: str,
        decimal: int = 3,
        hydrogen_mass: PositiveFloat = 1.007947,
        monolithic: bool = True,
        _merge_atom_types: bool = False,
    ):
        """
        Export this Interchange object to GROMACS files.

        Parameters
        ----------
        prefix: str
            The prefix to use for the GROMACS topology and coordinate files, i.e. "foo" will produce
            "foo.top", "foo.gro", and "foo_pointenergy.mdp".
        decimal: int, default=3
            The number of decimal places to use when writing the GROMACS coordinate file.
        hydrogen_mass : PositiveFloat, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters and is unsupported when virtual sites are present.
        monolithic: bool, default=False
            Whether the topology file should be monolithic (True) or reference individual .itp files (False). Note that
            these individual .itp files rely on ad hoc atom types and cannot be transferred between systems.
        _merge_atom_types: bool, default = False
            The flag to define behaviour of GROMACSWriter. If True, then similar atom types will be merged.
            If False, each atom will have its own atom type.

        Notes
        -----
        Molecule names in written files are not guaranteed to match the `Moleclue.name` attribute of the
        molecules in the topology, especially if they are empty strings or not unique.

        See :py:meth:`to_gro <Interchange.to_gro>` and :py:meth:`to_top <Interchange.to_top>` for more details.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        writer = GROMACSWriter(
            system=_convert(self, hydrogen_mass=hydrogen_mass),
            top_file=prefix + ".top",
            gro_file=prefix + ".gro",
        )

        writer.to_top(monolithic=monolithic, _merge_atom_types=_merge_atom_types)
        writer.to_gro(decimal=decimal)

        self.to_mdp(prefix + "_pointenergy.mdp")

    def to_mdp(self, file_path: Path | str):
        """
        Write a GROMACS run configuration ``.MDP`` file for a single-point energy.

        GROMACS considers many of the simulation parameters specified by an
        ``Interchange`` to be run configuration options rather than features of
        the topology. These options are set in the ``.MDP`` file. The written
        ``.MDP`` file includes the appropriate non-bonded configuration for the
        ``Interchange``. The ``nsteps``, ``nstenergy``, and ``continuation``
        configuration values are configured for a single-point energy
        calculation and may be changed as appropriate to perform other
        calculations. See the `GROMACS documentation`_ for details.

        .. _GROMACS documentation: https://manual.gromacs.org/documentation/\
        current/user-guide/mdp-options.html

        Parameters
        ----------
        file_path
            The path to the created GROMACS ``.MDP`` file

        """
        mdconfig = MDConfig.from_interchange(self)
        mdconfig.write_mdp_file(str(file_path))

    def to_top(
        self,
        file_path: Path | str,
        hydrogen_mass: PositiveFloat = 1.007947,
        monolithic: bool = True,
        _merge_atom_types: bool = False,
    ):
        """
        Export this Interchange to a GROMACS topology file.

        Parameters
        ----------
        file_path
            The path to the GROMACS topology file to write.
        hydrogen_mass : PositiveFloat, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters and is unsupported when virtual sites are present.
        monolithic: bool, default=False
            Whether the topology file should be monolithic (True) or reference individual .itp files (False). Note that
            these individual .itp files rely on ad hoc atom types and cannot be transferred between systems.
        _merge_atom_types: book, default=False
            The flag to define behaviour of GROMACSWriter. If True, then similar atom types will be merged.
            If False, each atom will have its own atom type.

        Notes
        -----
        Molecule names in written files are not guaranteed to match the `Moleclue.name` attribute of the
        molecules in the topology, especially if they are empty strings or not unique.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        GROMACSWriter(
            system=_convert(self, hydrogen_mass=hydrogen_mass),
            top_file=file_path,
            gro_file=tempfile.NamedTemporaryFile(suffix=".gro").file.name,
        ).to_top(
            monolithic=monolithic,
            _merge_atom_types=_merge_atom_types,
        )

    def to_gro(self, file_path: Path | str, decimal: int = 3):
        """
        Export this Interchange object to a GROMACS coordinate file.

        Parameters
        ----------
        file_path: Path | str
            The path to the GROMACS coordinate file to write.
        decimal: int, default=3
            The number of decimal places to use when writing the GROMACS coordinate file.

        Notes
        -----

        Residue IDs must be positive integers (or string representations thereof).

        Residue IDs greater than 99,999 are reduced modulo 100,000 in line with common GROMACS practice.

        Residue names and IDs from the topology are used, if present, and otherwise are generated internally.

        Behavior when some, but not all, residue names and IDs are present in the topology is undefined.

        If residue IDs are generated internally, they are assigned sequentially to each molecule starting at 1.

        """
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter
        from openff.interchange.smirnoff._gromacs import _convert

        # TODO: Write the coordinates without the full conversion
        GROMACSWriter(
            system=_convert(self),
            top_file=tempfile.NamedTemporaryFile(suffix=".top").file.name,
            gro_file=file_path,
        ).to_gro(decimal=decimal)

    def to_lammps(self, prefix: str, include_type_labels: bool = False):
        """
        Export this ``Interchange`` to LAMMPS data and run input files.

        Parameters
        ----------
        prefix: str
            The prefix to use for the LAMMPS data and run input files. For
            example, "foo" will produce files named "foo.lmp" and
            "foo_pointenergy.in".
        include_type_labels: bool
            If True, this will include the SMIRKS strings as LAMMPS type labels
            in the LAMMPS data file.
        """
        prefix = str(prefix)
        datafile_path = prefix + ".lmp"
        self.to_lammps_datafile(datafile_path, include_type_labels)
        self.to_lammps_input(
            prefix + "_pointenergy.in",
            datafile_path,
        )

    def to_lammps_datafile(self, file_path: Path | str, include_type_labels: bool = False):
        """Export this Interchange to a LAMMPS data file."""
        from openff.interchange.interop.lammps import to_lammps

        to_lammps(self, file_path, include_type_labels)

    def to_lammps_input(
        self,
        file_path: Path | str,
        data_file: Path | str | None = None,
    ):
        """
        Write a LAMMPS run input file for a single-point energy calculation.

        LAMMPS considers many of the simulation parameters specified by an
        ``Interchange`` to be run configuration options rather than features of
        the force field. These options are set in the run input file.

        Parameters
        ----------
        file_path
            The path to the created LAMMPS run input file
        data_file
            The path to the LAMMPS data file that should be read by the input
            file. If not given, ``file_path`` with the extension ``.lmp`` will
            be used.

        """
        if data_file is None:
            data_file = Path(file_path).with_suffix(".lmp")

        mdconfig = MDConfig.from_interchange(self)
        mdconfig.write_lammps_input(self, str(file_path), data_file=str(data_file))

    def to_openmm_system(
        self,
        combine_nonbonded_forces: bool = True,
        add_constrained_forces: bool = False,
        ewald_tolerance: float = 1e-4,
        hydrogen_mass: PositiveFloat = 1.007947,
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
        hydrogen_mass : PositiveFloat, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters.

        Returns
        -------
        system : openmm.System
            The OpenMM System object.

        Notes
        -----
        There are some sharp edges and quirks when using this method. Be aware of some documented
        issues in the :doc:`/using/edges` section of the user guide. If you encounter surprising
        behavior that is not documented, please raise an issue.

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

    def to_openmm(self, *args, **kwargs):
        return self.to_openmm_system(*args, **kwargs)

    def to_openmm_topology(
        self,
        collate: bool = False,
        ensure_unique_atom_names: str | bool = "residues",
    ):
        """
        Export components of this Interchange to an OpenMM Topology.

        Parameters
        ----------
        collate
            If ``False``, the default, all virtual sites will be added to a
            single residue at the end of the topology. If ``True``, virtual
            sites will be collated with their associated molecule and added to
            the residue of the last atom in the molecule they belong to.

        """
        from openff.interchange.interop.openmm._topology import to_openmm_topology

        return to_openmm_topology(
            self,
            collate=collate,
            ensure_unique_atom_names=ensure_unique_atom_names,
        )

    def to_openmm_simulation(
        self,
        integrator: "openmm.Integrator",
        combine_nonbonded_forces: bool = True,
        add_constrained_forces: bool = False,
        ewald_tolerance: float = 1e-4,
        hydrogen_mass: PositiveFloat = 1.007947,
        additional_forces: Iterable["openmm.Force"] = tuple(),
        **kwargs,
    ) -> "openmm.app.simulation.Simulation":
        """
        Export this Interchange to an OpenMM ``Simulation`` object.

        A :py:class:`Simulation <openmm.app.simulation.Simulation>` encapsulates
        all the information needed for a typical OpenMM simulation into a single
        object with a simple API.

        Positions are set on the ``Simulation`` if present on the
        ``Interchange``.

        Additional forces, such as a barostat, should be added with the
        ``additional_forces`` argument to avoid having to re-initialize
        the ``Context``. Re-initializing the ``Context`` after adding a
        ``Force`` is necessary due to `implementation details`_
        in OpenMM.

        .. _implementation details: https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#why-does-it-ignore-changes-i-make-to-a-system-or-force

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
        ewald_tolerance : float, default=1e-4
            The value passed to `NonbondedForce.setEwaldErrorTolerance`
        hydrogen_mass : PositiveFloat, default=1.007947
            The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
            than the default value, mass will be transferred from neighboring heavy atoms. Note that this is currently
            not applied to any waters.
        additional_forces : Iterable[openmm.Force], default=tuple()
            Additional forces to be added to the system, e.g. barostats, that are not
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

        Create an OpenMM simulation with a Langevin integrator and a Monte Carlo barostat:

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
        >>> simulation = interchange.to_openmm_simulation(  # doctest: +SKIP
        ...     integrator=integrator,
        ...     additional_forces=[barostat],
        ... )

        """
        import openmm.app

        system = self.to_openmm_system(
            combine_nonbonded_forces=combine_nonbonded_forces,
            add_constrained_forces=add_constrained_forces,
            ewald_tolerance=ewald_tolerance,
            hydrogen_mass=hydrogen_mass,
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
                self.get_positions(include_virtual_sites=True).to_openmm(),
            )

        return simulation

    @requires_package("openmm")
    def to_pdb(self, file_path: Path | str, include_virtual_sites: bool = False):
        """
        Export this Interchange to a .pdb file.

        Note that virtual sites are collated into each molecule, which differs from the default
        behavior of Interchange.to_openmm_topology.
        """
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
                collate=False,
                ensure_unique_atom_names="residues",
            )
            positions = get_positions_with_virtual_sites(self)

        else:
            openmm_topology = self.topology.to_openmm(
                ensure_unique_atom_names="residues",
            )
            positions = self.positions

        _to_pdb(file_path, openmm_topology, positions.to(unit.angstrom))

    def to_psf(self, file_path: Path | str):
        """Export this Interchange to a CHARMM-style .psf file."""
        raise UnsupportedExportError

    def to_crd(self, file_path: Path | str):
        """Export this Interchange to a CHARMM-style .crd file."""
        raise UnsupportedExportError

    def to_amber(
        self,
        prefix: str,
    ):
        """
        Export this Interchange object to Amber files.

        Parameters
        ----------
        prefix: str
            The prefix to use for the Amber parameter/topology, coordinate, and run files, i.e.
            "foo" will produce "foo.top", "foo.gro", and "foo_pointenergy.in".

        Notes
        -----
        The run input file is configured for a single-point energy calculation with sander. It is
        likely portable to pmemd with little or no work.

        """
        self.to_prmtop(f"{prefix}.prmtop")
        self.to_inpcrd(f"{prefix}.inpcrd")

        self.to_sander_input(f"{prefix}_pointenergy.in")

    def to_prmtop(self, file_path: Path | str):
        """Export this Interchange to an Amber .prmtop file."""
        from openff.interchange.interop.amber import to_prmtop

        to_prmtop(self, file_path)

    def to_inpcrd(self, file_path: Path | str):
        """Export this Interchange to an Amber .inpcrd file."""
        from openff.interchange.interop.amber import to_inpcrd

        to_inpcrd(self, file_path)

    def to_sander_input(self, file_path: Path | str):
        """
        Export this ``Interchange`` to a run input file for Amber's SANDER engine.

        Amber considers many of the simulation parameters specified by an
        ``Interchange`` to be run configuration options rather than parameters
        of the topology. These options are set in the SANDER or PMEMD run input
        file. The written SANDER input file includes the appropriate non-bonded
        configuration for the ``Interchange`` which are essential to reproduce
        the desired force field. The file also includes configuration for a
        single-point energy calculation, which should be modified to produce the
        desired simulation.
        """
        mdconfig = MDConfig.from_interchange(self)
        mdconfig.write_sander_input_file(str(file_path))

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
            >>> from foyer import Forcefield  # doctest: +SKIP
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> top = Topology.from_molecules([mol])
            >>> oplsaa = Forcefield(name="oplsaa")  # doctest: +SKIP
            >>> interchange = Interchange.from_foyer(topology=top, force_field=oplsaa)  # doctest: +SKIP
            >>> interchange  # doctest: +SKIP
            Interchange with 8 collections, non-periodic topology with 8 atoms.

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

        .. warning :: This method is experimental.

        Parameters
        ----------
        topology_file : Path | str
            The path to a GROMACS topology file.
        gro_file : Path | str
            The path to a GROMACS coordinate file.

        Returns
        -------
        interchange : Interchange
            An Interchange object representing the contents of the GROMACS files.

        Notes
        -----
        Bond parameters may not correctly be parsed, such as when using SMIRNOFF
        force fields with hydrogen bond constraints.

        """
        from openff.interchange.interop.gromacs._import._import import from_files
        from openff.interchange.interop.gromacs._interchange import to_interchange

        return to_interchange(
            from_files(
                top_file=topology_file,
                gro_file=gro_file,
            ),
        )

    def set_positions_from_gro(
        self,
        gro_file: Path | str,
    ):
        """
        Set the positions of this `Interchange` from a GROMACS coordinate `.gro` file.

        Only the coordinates from the `.gro` file are used. No effort is made to ensure the topologies are compatible
        with each other. This includes, for example, a lack of guarantee that the atom ordering in the `.gro` file
        matches the atom ordering in the `Interchange` object.

        `InvalidPositionsError` is raised if the number of rows in the coordinate array does not match the number of
        atoms in the topology of this `Interchange`.
        """
        from openff.interchange.interop.gromacs._import._import import _read_coordinates

        # should already be in nm, might not be necessary
        coordinates = _read_coordinates(gro_file).to(unit.nanometer)

        if coordinates.shape != (self.topology.n_atoms, 3):
            raise InvalidPositionsError(
                f"Coordinates in {gro_file} do not match the number of atoms in the topology. ",
                f"Parsed coordinates have shape {coordinates.shape} but topology has {self.topology.n_atoms} atoms.",
            )

        self.positions = coordinates

    @classmethod
    def from_openmm(
        cls,
        system: "openmm.System",
        topology: Union["openmm.app.Topology", Topology],
        positions: Quantity | None = None,
        box_vectors: Quantity | None = None,
    ) -> "Interchange":
        """
        Create an Interchange object from OpenMM objects.

        Notes
        -----
        If (topological) bonds in water are missing (physics) parameters, as is often the case with
        rigid water, these parameters will be filled in with values of 1 Angstrom equilibrium bond
        length and a default force constant of 50,000 kcal/mol/A^2, representing an arbitrarily
        stiff harmonic bond, and angle parameters of 104.5 degrees and 1.0 kcal/mol/rad^2,
        representing an arbitrarily harmonic angle. It is expected that these values will be
        overwritten by runtime MD options.

        See more limitations and sharp edges in the user guide: https://docs.openforcefield.org/projects/interchange/en/latest/using/edges.html#quirks-of-from-openmm

        Parameters
        ----------
        system : openmm.System
            The OpenMM system.
        topology : openmm.app.Topology
            The OpenMM topology.
        positions : openmm.unit.Quantity or openff.units.Quantity, optional
            The positions of particles in this system and/or topology.
        box_vectors : openmm.unit.Quantity or openff.units.Quantity, optional
            The vectors of the simulation box associated with this system and/or topology.

        Returns
        -------
        interchange : Interchange
            An Interchange object representing the contents of the OpenMM objects.

        Notes
        -----
        An `openmm.CMMotionRemover` force, if present, is ignored.

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
    def __getitem__(
        self,
        item: Literal["GBSA"],
    ) -> "SMIRNOFFGBSACollection": ...

    @overload
    def __getitem__(
        self,
        item: str,
    ) -> "Collection": ...

    def __getitem__(self, item: str):
        """Syntax sugar for looking up collections or other components."""
        if type(item) is not str:
            raise LookupError(
                f"Only str arguments can be currently be used for lookups.\nFound item {item} of type {type(item)}",
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

    def __add__(self, other: "Interchange") -> "Interchange":
        """Combine two Interchange objects."""
        warnings.warn(
            "The `+` operator is deprecated. Use `Interchange.combine` instead.",
            InterchangeDeprecationWarning,
        )

        return self.combine(other)

    def combine(self, other: "Interchange") -> "Interchange":
        """Combine two Interchange objects."""
        from openff.interchange.operations._combine import _combine

        return _combine(self, other)

    def __repr__(self) -> str:
        periodic = self.box is not None
        n_atoms = self.topology.n_atoms
        return (
            f"Interchange with {len(self.collections)} collections, "
            f"{'' if periodic else 'non-'}periodic topology with {n_atoms} atoms."
        )
