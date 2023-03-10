"""Interfaces with OpenMM."""
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import openmm
from openmm import unit

from openff.interchange.exceptions import (
    PluginCompatibilityError,
    UnsupportedImportError,
)
from openff.interchange.interop.openmm._positions import to_openmm_positions
from openff.interchange.interop.openmm._topology import to_openmm_topology

__all__ = [
    "to_openmm",
    "to_openmm_topology",
    "to_openmm_positions",
    "from_openmm",
]

if TYPE_CHECKING:
    import openmm.app
    from openff.toolkit.topology import Topology

    from openff.interchange.smirnoff._nonbonded import (
        SMIRNOFFElectrostaticsCollection,
        SMIRNOFFvdWCollection,
    )
    from openff.interchange.smirnoff._valence import (
        SMIRNOFFAngleCollection,
        SMIRNOFFBondCollection,
        SMIRNOFFProperTorsionCollection,
    )


def to_openmm(
    interchange,
    combine_nonbonded_forces: bool = False,
    add_constrained_forces: bool = False,
) -> openmm.System:
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

    Returns
    -------
    openmm_sys : openmm.System
        The corresponding OpenMM System object

    """
    from openff.units import unit as off_unit

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
            try:
                collection.check_openmm_requirements(combine_nonbonded_forces)
            except AssertionError as error:
                raise PluginCompatibilityError(
                    f"Collection of type {type(collection)} failed a compatibility check.",
                ) from error

    openmm_sys = openmm.System()

    if interchange.box is not None:
        box = interchange.box.m_as(off_unit.nanometer)
        openmm_sys.setDefaultPeriodicBoxVectors(*box)

    particle_map = _process_nonbonded_forces(
        interchange,
        openmm_sys,
        combine_nonbonded_forces=combine_nonbonded_forces,
    )

    constrained_pairs = _process_constraints(interchange, openmm_sys, particle_map)

    _process_torsion_forces(interchange, openmm_sys, particle_map)
    _process_improper_torsion_forces(interchange, openmm_sys, particle_map)
    _process_angle_forces(
        interchange,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
        particle_map=particle_map,
    )
    _process_bond_forces(
        interchange,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
        particle_map=particle_map,
    )

    for collection in interchange.collections.values():
        if collection.is_plugin:
            try:
                collection.modify_openmm_forces(
                    interchange,
                    openmm_sys,
                    add_constrained_forces=add_constrained_forces,
                    constrained_pairs=constrained_pairs,
                    particle_map=particle_map,
                )
            except NotImplementedError:
                continue

    return openmm_sys


def from_openmm(
    topology: Optional["openmm.app.Topology"] = None,
    system: Optional[openmm.System] = None,
    positions=None,
    box_vectors=None,
):
    """Create an Interchange object from OpenMM data."""
    import warnings

    from openff.interchange import Interchange

    warnings.warn(
        "Importing from OpenMM `System` objects is EXPERIMENTAL, fragile, and "
        "currently unlikely to produce expected results for all but the simplest "
        "use cases. It is thereforce currently unsuitable for production work. "
        "However, it is an area of active development; if this function would "
        "enable key components of your workflows, feedback is welcome. Please "
        'file an issue or create a new discussion (see the "Discussions" tab.',
    )

    interchange = Interchange()

    if system:
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                vdw, coul = _convert_nonbonded_force(force)
                interchange.collections["vdW"] = vdw
                interchange.collections["Electrostatics"] = coul
            elif isinstance(force, openmm.HarmonicBondForce):
                bonds = _convert_harmonic_bond_force(force)
                interchange.collections["Bonds"] = bonds
            elif isinstance(force, openmm.HarmonicAngleForce):
                angles = _convert_harmonic_angle_force(force)
                interchange.collections["Angles"] = angles
            elif isinstance(force, openmm.PeriodicTorsionForce):
                proper_torsions = _convert_periodic_torsion_force(force)
                interchange.collections["ProperTorsions"] = proper_torsions
            elif isinstance(force, openmm.CMMotionRemover):
                pass
            else:
                raise UnsupportedImportError(
                    f"Unsupported OpenMM Force type ({type(force)}) found.",
                )

    if topology is not None:
        from openff.interchange.components.toolkit import _simple_topology_from_openmm

        openff_topology = _simple_topology_from_openmm(topology)

        interchange.topology = openff_topology

    if positions is not None:
        interchange.positions = positions

    if box_vectors is not None:
        interchange.box = box_vectors

    return interchange


def _convert_nonbonded_force(
    force: openmm.NonbondedForce,
) -> Tuple["SMIRNOFFvdWCollection", "SMIRNOFFElectrostaticsCollection"]:
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, TopologyKey
    from openff.interchange.smirnoff._nonbonded import (
        SMIRNOFFElectrostaticsCollection,
        SMIRNOFFvdWCollection,
    )

    if force.getNonbondedMethod() != 0:
        raise UnsupportedImportError(
            "Importing from OpenMM only currently supported with `openmm.NonbondedForce.PME`.",
        )

    vdw = SMIRNOFFvdWCollection()
    electrostatics = SMIRNOFFElectrostaticsCollection(version=0.4, scale_14=0.833333)

    n_parametrized_particles = force.getNumParticles()

    for idx in range(n_parametrized_particles):
        charge, sigma, epsilon = force.getParticleParameters(idx)
        top_key = TopologyKey(atom_indices=(idx,))
        pot_key = PotentialKey(id=f"{idx}")
        pot = Potential(
            parameters={
                "sigma": from_openmm_quantity(sigma),
                "epsilon": from_openmm_quantity(epsilon),
            },
        )
        vdw.key_map.update({top_key: pot_key})
        vdw.potentials.update({pot_key: pot})

        electrostatics.key_map.update({top_key: pot_key})
        electrostatics.potentials.update(
            {pot_key: Potential(parameters={"charge": from_openmm_quantity(charge)})},
        )

    # 0 == openmm.NonbondedForce.PME:
    if force.getNonbondedMethod() == 0:
        electrostatics.periodic_potential = "Ewald3D-ConductingBoundary"
        vdw.method = "cutoff"
        vdw.cutoff = force.getCutoffDistance()
    else:
        raise UnsupportedImportError(
            f"Parsing a non-bonded force of type {type(force)} with {force.getNonbondedMethod()} not yet supported.",
        )

    return vdw, electrostatics


def _convert_harmonic_bond_force(
    force: openmm.HarmonicBondForce,
) -> "SMIRNOFFBondCollection":
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import BondKey, PotentialKey
    from openff.interchange.smirnoff._valence import SMIRNOFFBondCollection

    bonds = SMIRNOFFBondCollection()

    n_parametrized_bonds = force.getNumBonds()

    for idx in range(n_parametrized_bonds):
        atom1, atom2, length, k = force.getBondParameters(idx)
        top_key = BondKey(atom_indices=(atom1, atom2))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}")
        pot = Potential(
            parameters={
                "length": from_openmm_quantity(length),
                "k": from_openmm_quantity(k),
            },
        )

        bonds.key_map.update({top_key: pot_key})
        bonds.potentials.update({pot_key: pot})

    return bonds


def _convert_harmonic_angle_force(
    force: openmm.HarmonicAngleForce,
) -> "SMIRNOFFAngleCollection":
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import AngleKey, PotentialKey
    from openff.interchange.smirnoff._valence import SMIRNOFFAngleCollection

    angles = SMIRNOFFAngleCollection()

    n_parametrized_angles = force.getNumAngles()

    for idx in range(n_parametrized_angles):
        atom1, atom2, atom3, angle, k = force.getAngleParameters(idx)
        top_key = AngleKey(atom_indices=(atom1, atom2, atom3))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}")
        pot = Potential(
            parameters={
                "angle": from_openmm_quantity(angle),
                "k": from_openmm_quantity(k),
            },
        )

        angles.key_map.update({top_key: pot_key})
        angles.potentials.update({pot_key: pot})

    return angles


def _convert_periodic_torsion_force(
    force: openmm.PeriodicTorsionForce,
) -> "SMIRNOFFProperTorsionCollection":
    # TODO: Can impropers be separated out from a PeriodicTorsionForce?
    # Maybe by seeing if a quartet is in mol/top.propers or .impropers
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, ProperTorsionKey
    from openff.interchange.smirnoff._valence import SMIRNOFFProperTorsionCollection

    proper_torsions = SMIRNOFFProperTorsionCollection()

    n_parametrized_torsions = force.getNumTorsions()

    for idx in range(n_parametrized_torsions):
        atom1, atom2, atom3, atom4, per, phase, k = force.getTorsionParameters(idx)
        # TODO: Process layered torsions
        # TODO: Check if this torsion is an improper
        top_key = ProperTorsionKey(atom_indices=(atom1, atom2, atom3, atom4), mult=0)
        while top_key in proper_torsions.key_map:
            top_key.mult = top_key.mult + 1  # type: ignore[operator]

        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}-{atom4}", mult=top_key.mult)
        pot = Potential(
            parameters={
                "periodicity": int(per) * unit.dimensionless,
                "phase": from_openmm_quantity(phase),
                "k": from_openmm_quantity(k),
                "idivf": 1 * unit.dimensionless,
            },
        )

        proper_torsions.key_map.update({top_key: pot_key})
        proper_torsions.potentials.update({pot_key: pot})

    return proper_torsions


def _to_pdb(file_path: Union[Path, str], topology: "Topology", positions):
    from openff.units.openmm import to_openmm
    from openmm import app

    openmm_topology = topology.to_openmm(ensure_unique_atom_names=False)

    positions = to_openmm(positions)

    with open(file_path, "w") as outfile:
        app.PDBFile.writeFile(openmm_topology, positions, outfile)


def _get_nonbonded_force_from_openmm_system(
    system: openmm.System,
) -> openmm.NonbondedForce:
    """Get a single NonbondedForce object with an OpenMM System."""
    for force in system.getForces():
        if type(force) == openmm.NonbondedForce:
            return force


def _get_partial_charges_from_openmm_system(system: openmm.System) -> List[float]:
    """Get partial charges from an OpenMM interchange as a unit.Quantity array."""
    # TODO: deal with virtual sites
    n_particles = system.getNumParticles()
    force = _get_nonbonded_force_from_openmm_system(system)
    # TODO: don't assume the partial charge will always be parameter 0
    # partial_charges = [openmm_to_pint(force.getParticleParameters(idx)[0]) for idx in range(n_particles)]
    partial_charges = [
        force.getParticleParameters(idx)[0] / unit.elementary_charge
        for idx in range(n_particles)
    ]

    return partial_charges
