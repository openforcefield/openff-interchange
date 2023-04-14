from typing import TYPE_CHECKING, Optional, Tuple

import openmm

from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ProperTorsionCollection,
)
from openff.interchange.exceptions import UnsupportedImportError

if TYPE_CHECKING:
    import openmm.app


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
        'file an issue or create a new discussion (see the "Discussions" tab).',
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
) -> Tuple["vdWCollection", "ElectrostaticsCollection"]:
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._nonbonded import (
        ElectrostaticsCollection,
        vdWCollection,
    )
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, TopologyKey

    if force.getNonbondedMethod() != 4:
        raise UnsupportedImportError(
            "Importing from OpenMM only currently supported with `openmm.NonbondedForce.PME`.",
        )

    vdw = vdWCollection()
    electrostatics = ElectrostaticsCollection(version=0.4, scale_14=0.833333)

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

    if force.getNonbondedMethod() == 4:
        vdw.cutoff = force.getCutoffDistance()
    else:
        raise UnsupportedImportError(
            f"Parsing a non-bonded force of type {type(force)} with {force.getNonbondedMethod()} not yet supported.",
        )

    return vdw, electrostatics


def _convert_harmonic_bond_force(
    force: openmm.HarmonicBondForce,
) -> "BondCollection":
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import BondCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import BondKey, PotentialKey

    bonds = BondCollection()

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
) -> "AngleCollection":
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import AngleCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import AngleKey, PotentialKey

    angles = AngleCollection()

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
) -> "ProperTorsionCollection":
    # TODO: Can impropers be separated out from a PeriodicTorsionForce?
    # Maybe by seeing if a quartet is in mol/top.propers or .impropers
    from openff.units import unit
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import ProperTorsionCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, ProperTorsionKey

    proper_torsions = ProperTorsionCollection()

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
