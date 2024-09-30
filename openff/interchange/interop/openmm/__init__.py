"""Interfaces with OpenMM."""

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from openff.utilities.utilities import has_package, requires_package

from openff.interchange._annotations import PositiveFloat
from openff.interchange.exceptions import (
    NegativeMassError,
    PluginCompatibilityError,
    UnsupportedExportError,
)
from openff.interchange.interop.openmm._import._import import from_openmm
from openff.interchange.interop.openmm._positions import to_openmm_positions
from openff.interchange.interop.openmm._topology import to_openmm_topology
from openff.interchange.smirnoff._base import SMIRNOFFCollection

if has_package("openmm"):
    import openmm
    import openmm.app

if TYPE_CHECKING:
    from openff.interchange import Interchange

__all__ = [
    "to_openmm",
    "to_openmm_topology",
    "to_openmm_positions",
    "from_openmm",
]


@requires_package("openmm")
def to_openmm_system(
    interchange: "Interchange",
    combine_nonbonded_forces: bool = False,
    add_constrained_forces: bool = False,
    ewald_tolerance: float = 1e-4,
    hydrogen_mass: PositiveFloat = 1.007947,
) -> "openmm.System":
    """
    Convert an Interchange to an OpenmM System.

    Parameters
    ----------
    interchange : openff.interchange.Interchange
        An OpenFF Interchange object
    combine_nonbonded_forces : bool, default=False
        If True, an attempt will be made to combine all non-bonded interactions into a single openmm.NonbondedForce.
        If False, non-bonded interactions will be split across multiple forces.
    add_constrained_forces : bool, default=False,
        If True, add valence forces that might be overridden by constraints, i.e. call `addBond` or `addAngle`
        on a bond or angle that is fully constrained.
    ewald_tolerance : float, default=1e-4
        The value passed to `NonbondedForce.setEwaldErrorTolerance`
    hydrogen_mass : PositiveFloat, default=1.007947
        The mass to use for hydrogen atoms if not present in the topology. If non-trivially different
        than the default value, mass will be transferred from neighboring heavy atoms.

    Returns
    -------
    system : openmm.System
        The corresponding OpenMM System object

    """
    from openff.toolkit import unit as off_unit

    from openff.interchange.interop.openmm._gbsa import _process_gbsa
    from openff.interchange.interop.openmm._nonbonded import _process_nonbonded_forces
    from openff.interchange.interop.openmm._valence import (
        _process_angle_forces,
        _process_bond_forces,
        _process_constraints,
        _process_improper_torsion_forces,
        _process_torsion_forces,
    )

    for collection in interchange.collections.values():
        if collection.is_plugin:
            assert isinstance(collection, SMIRNOFFCollection)

            try:
                collection.check_openmm_requirements(combine_nonbonded_forces)
            except AssertionError as error:
                raise PluginCompatibilityError(
                    f"Collection of type {type(collection)} failed a compatibility check.",
                ) from error

    system = openmm.System()

    if interchange.box is not None:
        box = interchange.box.m_as(off_unit.nanometer)
        system.setDefaultPeriodicBoxVectors(*box)

    particle_map = _process_nonbonded_forces(
        interchange,
        system,
        combine_nonbonded_forces=combine_nonbonded_forces,
        ewald_tolerance=ewald_tolerance,
    )

    constrained_pairs = _process_constraints(interchange, system, particle_map)

    _process_torsion_forces(interchange, system, particle_map)
    _process_improper_torsion_forces(interchange, system, particle_map)
    _process_angle_forces(
        interchange,
        system,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
        particle_map=particle_map,
    )
    _process_bond_forces(
        interchange,
        system,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
        particle_map=particle_map,
    )

    _process_gbsa(
        interchange,
        system,
    )

    # TODO: Apply HMR before or after final plugin touch point?
    _apply_hmr(
        system,
        interchange,
        hydrogen_mass=hydrogen_mass,
    )

    for collection in interchange.collections.values():
        if collection.is_plugin:
            assert isinstance(collection, SMIRNOFFCollection)

            try:
                collection.modify_openmm_forces(
                    interchange,
                    system,
                    add_constrained_forces=add_constrained_forces,
                    constrained_pairs=constrained_pairs,
                    particle_map=particle_map,
                )
            except NotImplementedError:
                continue

    return system


to_openmm = to_openmm_system


@requires_package("openmm")
def _to_pdb(
    file_path: Path | str | TextIO,
    topology: "openmm.app.Topology",
    positions,
):
    from openff.units.openmm import ensure_quantity

    # Deal with the possibility of `StringIO`
    manager: nullcontext[TextIO] | TextIO  # MyPy needs some help here
    if isinstance(file_path, (str, Path)):
        manager = open(file_path, "w")
    else:
        manager = nullcontext(file_path)

    with manager as outfile:
        openmm.app.PDBFile.writeFile(
            topology=topology,
            positions=ensure_quantity(positions, "openmm"),
            file=outfile,
        )


@requires_package("openmm")
def _apply_hmr(
    system: "openmm.System",
    interchange: "Interchange",
    hydrogen_mass: PositiveFloat,
):
    from openff.toolkit import Molecule

    if abs(hydrogen_mass - 1.008) < 1e-3:
        return

    if system.getNumParticles() != interchange.topology.n_atoms:
        raise UnsupportedExportError(
            "Hydrogen mass repartitioning with virtual sites present, even on rigid water, is not yet supported.",
        )

    water = Molecule.from_smiles("O")

    def _is_water(molecule: Molecule) -> bool:
        return molecule.is_isomorphic_with(water)

    _hydrogen_mass = hydrogen_mass * openmm.unit.dalton

    for bond in interchange.topology.bonds:
        heavy_atom, hydrogen_atom = bond.atoms

        if heavy_atom.atomic_number == 1:
            heavy_atom, hydrogen_atom = hydrogen_atom, heavy_atom

        # TODO: This should only skip rigid waters, even though HMR or flexible water is questionable
        if (
            (hydrogen_atom.atomic_number == 1)
            and (heavy_atom.atomic_number != 1)
            and not (_is_water(hydrogen_atom.molecule))
        ):
            hydrogen_index = interchange.topology.atom_index(hydrogen_atom)
            heavy_index = interchange.topology.atom_index(heavy_atom)
            heavy_mass = system.getParticleMass(heavy_index)

            # This will need to be wired up through the OpenFF-OpenMM particle index map
            # when virtual sites + HMR are supported
            mass_to_transfer = _hydrogen_mass - system.getParticleMass(hydrogen_index)

            if mass_to_transfer > heavy_mass:
                raise NegativeMassError(
                    f"Particle with index {heavy_index} would have a negative mass after hydrogen "
                    "mass repartitioning. Consider transferring a smaller mass than "
                    f"{hydrogen_mass=}.",
                )

            system.setParticleMass(
                hydrogen_index,
                hydrogen_mass * openmm.unit.dalton,
            )

            system.setParticleMass(
                heavy_index,
                heavy_mass - mass_to_transfer,
            )
