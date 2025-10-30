"""
Helper functions for producing `openmm.Force` objects for valence terms.
"""

import functools

from openff.units.openmm import to_openmm as to_openmm_quantity
from openff.utilities.utilities import has_package

from openff.interchange.exceptions import UnsupportedExportError
from openff.interchange.models import BaseVirtualSiteKey, PotentialKey
from openff.interchange.smirnoff._valence import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
    SMIRNOFFImproperTorsionCollection,
    SMIRNOFFProperTorsionCollection,
)

if has_package("openmm"):
    import openmm


def _process_constraints(
    interchange,
    openmm_sys,
    particle_map: dict[int | BaseVirtualSiteKey, int],
) -> set[tuple[int, ...]]:
    """
    Process the Constraints section of an Interchange object.
    """
    try:
        constraint_handler = interchange["Constraints"]
    except LookupError:
        return set()

    constrained_pairs: set[tuple[int, ...]] = set()

    for top_key, pot_key in constraint_handler.key_map.items():
        openff_indices = top_key.atom_indices

        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        params = constraint_handler.potentials[pot_key].parameters
        distance = params["distance"]
        distance_omm = distance.m_as("nanometer")

        constrained_pairs.add(tuple(sorted(openmm_indices)))
        openmm_sys.addConstraint(
            openmm_indices[0],
            openmm_indices[1],
            distance_omm,
        )

    return constrained_pairs


@functools.lru_cache
def _lookup_bond_parameters_from_potential_key(
    collection: SMIRNOFFBondCollection,
    potential_key: PotentialKey,
) -> tuple[float, float]:
    parameters = collection.potentials[potential_key].parameters

    return (
        parameters["k"].m_as("kilojoule / nanometer ** 2 / mole"),
        parameters["length"].m_as("nanometer"),
    )


def _process_bond_forces(
    interchange,
    openmm_sys,
    add_constrained_forces: bool,
    constrained_pairs: set[tuple[int, ...]],
    particle_map: dict[int | BaseVirtualSiteKey, int],
):
    """
    Process the Bonds section of an Interchange object.
    """
    try:
        bond_handler = interchange.collections["Bonds"]
    except KeyError:
        return

    if bond_handler.expression == "k/2*(r-length)**2":
        harmonic_bond_force = openmm.HarmonicBondForce()
    else:
        raise UnsupportedExportError(
            f"Found an unsupported functional form in the bond handler:\n\t{bond_handler.expression=}",
        )

    openmm_sys.addForce(harmonic_bond_force)

    has_constraint_handler = "Constraints" in interchange.collections

    for top_key, pot_key in bond_handler.key_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        if has_constraint_handler and not add_constrained_forces:
            if _is_constrained(
                constrained_pairs,
                (openmm_indices[0], openmm_indices[1]),
            ):
                # This bond's length is constrained, dpo so not add a bond force
                continue

        k, length = _lookup_bond_parameters_from_potential_key(bond_handler, pot_key)

        harmonic_bond_force.addBond(
            particle1=openmm_indices[0],
            particle2=openmm_indices[1],
            length=length,
            k=k,
        )


@functools.lru_cache
def _lookup_angle_parameters_from_potential_key(
    collection: SMIRNOFFAngleCollection,
    potential_key: PotentialKey,
) -> tuple[float, float]:
    parameters = collection.potentials[potential_key].parameters

    return (
        parameters["k"].m_as("kilojoule / radian / mole"),
        parameters["angle"].m_as("radian"),
    )


def _process_angle_forces(
    interchange,
    openmm_sys,
    add_constrained_forces: bool,
    constrained_pairs: set[tuple[int, ...]],
    particle_map,
):
    """
    Process the Angles section of an Interchange object.
    """
    try:
        angle_handler = interchange.collections["Angles"]
    except KeyError:
        return

    if angle_handler.expression == "k/2*(theta-angle)**2":
        custom = False
        harmonic_angle_force = openmm.HarmonicAngleForce()
    elif angle_handler.expression == "k/2*(cos(theta)-cos(angle))**2":
        custom = True
        harmonic_angle_force = openmm.CustomAngleForce(
            angle_handler.expression.replace("**", "^"),
        )
        for parameter_name in angle_handler.potential_parameters():
            harmonic_angle_force.addPerAngleParameter(parameter_name)
    else:
        raise UnsupportedExportError(
            f"Found an unsupported functional form in the angle handler:\n\t{angle_handler.expression=}",
        )

    openmm_sys.addForce(harmonic_angle_force)

    has_constraint_handler = "Constraints" in interchange.collections

    for top_key, pot_key in angle_handler.key_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        if has_constraint_handler and not add_constrained_forces:
            if _is_constrained(
                constrained_pairs,
                (openmm_indices[0], openmm_indices[2]),
            ):
                if _is_constrained(
                    constrained_pairs,
                    (openmm_indices[0], openmm_indices[1]),
                ):
                    if _is_constrained(
                        constrained_pairs,
                        (openmm_indices[1], openmm_indices[2]),
                    ):
                        # This angle's geometry is fully subject to constraints, so do
                        # not an angle force
                        continue

        if custom:
            params = angle_handler.potentials[pot_key].parameters
            parameter_values = [to_openmm_quantity(params[val]) for val in angle_handler.potential_parameters()]

            harmonic_angle_force.addAngle(
                openmm_indices[0],
                openmm_indices[1],
                openmm_indices[2],
                parameter_values,
            )

        else:
            k, angle = _lookup_angle_parameters_from_potential_key(angle_handler, pot_key)

            harmonic_angle_force.addAngle(
                particle1=openmm_indices[0],
                particle2=openmm_indices[1],
                particle3=openmm_indices[2],
                angle=angle,
                k=k,
            )


def _process_torsion_forces(interchange, openmm_sys, particle_map):
    if "ProperTorsions" in interchange.collections:
        _process_proper_torsion_forces(interchange, openmm_sys, particle_map)
    if "RBTorsions" in interchange.collections:
        _process_rb_torsion_forces(interchange, openmm_sys, particle_map)


@functools.lru_cache
def _lookup_torsion_parameters_from_potential_key(
    collection: SMIRNOFFProperTorsionCollection | SMIRNOFFImproperTorsionCollection,
    potential_key: PotentialKey,
) -> tuple[float, int, float, int]:
    parameters = collection.potentials[potential_key].parameters

    # float rounding can sometimes make Quantity(1) into 0.9999999999999999
    return (
        parameters["k"].m_as("kilojoule / mole"),
        int(round(parameters["periodicity"], 10)),
        parameters["phase"].m_as("radian"),
        int(round(parameters["idivf"], 10)),
    )


def _process_proper_torsion_forces(interchange, openmm_sys, particle_map):
    """
    Process the Propers section of an Interchange object.
    """
    torsion_force = openmm.PeriodicTorsionForce()
    openmm_sys.addForce(torsion_force)

    proper_torsion_handler = interchange["ProperTorsions"]

    for top_key, pot_key in proper_torsion_handler.key_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        k, periodicity, phase, idivf = _lookup_torsion_parameters_from_potential_key(proper_torsion_handler, pot_key)

        if idivf == 0:
            raise RuntimeError("Found an idivf of 0.")

        torsion_force.addTorsion(
            openmm_indices[0],
            openmm_indices[1],
            openmm_indices[2],
            openmm_indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _process_rb_torsion_forces(interchange, openmm_sys, particle_map):
    """
    Process Ryckaert-Bellemans torsions.
    """
    rb_force = openmm.RBTorsionForce()
    openmm_sys.addForce(rb_force)

    rb_torsion_handler = interchange["RBTorsions"]

    for top_key, pot_key in rb_torsion_handler.key_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        params = rb_torsion_handler.potentials[pot_key].parameters

        c0 = params["c0"].m_as("kilojoule / mole")
        c1 = params["c1"].m_as("kilojoule / mole")
        c2 = params["c2"].m_as("kilojoule / mole")
        c3 = params["c3"].m_as("kilojoule / mole")
        c4 = params["c4"].m_as("kilojoule / mole")
        c5 = params["c5"].m_as("kilojoule / mole")

        rb_force.addTorsion(
            openmm_indices[0],
            openmm_indices[1],
            openmm_indices[2],
            openmm_indices[3],
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        )


def _process_improper_torsion_forces(interchange, openmm_sys, particle_map):
    """
    Process the Impropers section of an Interchange object.
    """
    if "ImproperTorsions" not in interchange.collections.keys():
        return

    for force in openmm_sys.getForces():
        if type(force) is openmm.PeriodicTorsionForce:
            torsion_force = force
            break
    else:
        torsion_force = openmm.PeriodicTorsionForce()

    improper_torsion_handler = interchange["ImproperTorsions"]

    for top_key, pot_key in improper_torsion_handler.key_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = tuple(particle_map[index] for index in openff_indices)

        k, periodicity, phase, idivf = _lookup_torsion_parameters_from_potential_key(improper_torsion_handler, pot_key)

        torsion_force.addTorsion(
            openmm_indices[0],
            openmm_indices[1],
            openmm_indices[2],
            openmm_indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _is_constrained(
    constrained_pairs: set[tuple[int, ...]],
    pair: tuple[int, int],
) -> bool:
    if (pair[0], pair[1]) in constrained_pairs:
        return True
    if (pair[1], pair[0]) in constrained_pairs:
        return True
    return False
