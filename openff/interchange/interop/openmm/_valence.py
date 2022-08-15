from typing import Dict, List, Tuple

import openmm
from openff.units import unit as off_unit
from openff.units.openmm import to_openmm as to_openmm_quantity

from openff.interchange.exceptions import UnsupportedExportError


def _process_constraints(openff_sys, openmm_sys, particle_map: Dict[int, int]):
    """
    Process the Constraints section of an Interchange object.
    """
    try:
        constraint_handler = openff_sys["Constraints"]
    except LookupError:
        return

    constrained_pairs = list()

    for top_key, pot_key in constraint_handler.slot_map.items():
        openff_indices = top_key.atom_indices
        openmm_indices = (
            particle_map[openff_indices[0]],
            particle_map[openff_indices[1]],
        )
        params = constraint_handler.constraints[pot_key].parameters
        distance = params["distance"]
        distance_omm = distance.m_as(off_unit.nanometer)

        constrained_pairs.append(tuple(sorted(openmm_indices)))
        openmm_sys.addConstraint(
            openmm_indices[0],
            openmm_indices[1],
            distance_omm,
        )

    return constrained_pairs


def _process_bond_forces(
    openff_sys,
    openmm_sys,
    add_constrained_forces: bool,
    constrained_pairs: List[Tuple[int]],
):
    """
    Process the Bonds section of an Interchange object.
    """
    try:
        bond_handler = openff_sys.handlers["Bonds"]
    except KeyError:
        return

    if bond_handler.expression == "k/2*(r-length)**2":
        harmonic_bond_force = openmm.HarmonicBondForce()
    else:
        raise UnsupportedExportError(
            "Found an unsupported functional form in the bond handler:\n\t"
            f"{bond_handler.expression=}"
        )

    openmm_sys.addForce(harmonic_bond_force)

    has_constraint_handler = "Constraints" in openff_sys.handlers

    for top_key, pot_key in bond_handler.slot_map.items():

        indices = top_key.atom_indices

        if has_constraint_handler and not add_constrained_forces:
            if _is_constrained(constrained_pairs, (indices[0], indices[1])):
                # This bond's length is constrained, dpo so not add a bond force
                continue

        indices = top_key.atom_indices
        params = bond_handler.potentials[pot_key].parameters
        k = params["k"].m_as(
            off_unit.kilojoule / off_unit.nanometer**2 / off_unit.mol
        )
        length = params["length"].m_as(off_unit.nanometer)

        harmonic_bond_force.addBond(
            particle1=indices[0],
            particle2=indices[1],
            length=length,
            k=k,
        )


def _process_angle_forces(
    openff_sys,
    openmm_sys,
    add_constrained_forces: bool,
    constrained_pairs: List[Tuple[int]],
):
    """
    Process the Angles section of an Interchange object.
    """
    try:
        angle_handler = openff_sys.handlers["Angles"]
    except KeyError:
        return

    if angle_handler.expression == "k/2*(theta-angle)**2":
        custom = False
        harmonic_angle_force = openmm.HarmonicAngleForce()
    elif angle_handler.expression == "k/2*(cos(theta)-cos(angle))**2":
        custom = True
        harmonic_angle_force = openmm.CustomAngleForce(
            angle_handler.expression.replace("**", "^")
        )
        for parameter_name in angle_handler._potential_parameters():
            harmonic_angle_force.addPerAngleParameter(parameter_name)
    else:
        raise UnsupportedExportError(
            "Found an unsupported functional form in the angle handler:\n\t"
            f"{angle_handler.expression=}"
        )

    openmm_sys.addForce(harmonic_angle_force)

    has_constraint_handler = "Constraints" in openff_sys.handlers

    for top_key, pot_key in angle_handler.slot_map.items():

        indices = top_key.atom_indices

        if has_constraint_handler and not add_constrained_forces:
            if _is_constrained(constrained_pairs, (indices[0], indices[2])):
                if _is_constrained(constrained_pairs, (indices[0], indices[1])):
                    if _is_constrained(constrained_pairs, (indices[1], indices[2])):
                        # This angle's geometry is fully subject to constraints, so do
                        # not an angle force
                        continue

        if custom:
            params = angle_handler.potentials[pot_key].parameters
            parameter_values = [
                to_openmm_quantity(params[val])
                for val in angle_handler._potential_parameters()
            ]

            harmonic_angle_force.addAngle(
                indices[0],
                indices[1],
                indices[2],
                parameter_values,
            )

        else:
            params = angle_handler.potentials[pot_key].parameters
            k = params["k"].m_as(off_unit.kilojoule / off_unit.rad / off_unit.mol)
            angle = params["angle"].m_as(off_unit.radian)

            harmonic_angle_force.addAngle(
                particle1=indices[0],
                particle2=indices[1],
                particle3=indices[2],
                angle=angle,
                k=k,
            )


def _process_torsion_forces(openff_sys, openmm_sys):
    if "ProperTorsions" in openff_sys.handlers:
        _process_proper_torsion_forces(openff_sys, openmm_sys)
    if "RBTorsions" in openff_sys.handlers:
        _process_rb_torsion_forces(openff_sys, openmm_sys)


def _process_proper_torsion_forces(openff_sys, openmm_sys):
    """
    Process the Propers section of an Interchange object.
    """
    torsion_force = openmm.PeriodicTorsionForce()
    openmm_sys.addForce(torsion_force)

    proper_torsion_handler = openff_sys["ProperTorsions"]

    for top_key, pot_key in proper_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = proper_torsion_handler.potentials[pot_key].parameters

        k = params["k"].m_as(off_unit.kilojoule / off_unit.mol)
        periodicity = int(params["periodicity"])
        phase = params["phase"].m_as(off_unit.radian)
        # Work around a pint gotcha:
        # >>> import pint
        # >>> u = pint.UnitRegistry()
        # >>> val
        # <Quantity(1.0, 'dimensionless')>
        # >>> val.m
        # 0.9999999999
        # >>> int(val)
        # 0
        # >>> int(round(val, 0))
        # 1
        # >>> round(val.m_as(u.dimensionless), 0)
        # 1.0
        # >>> round(val, 0).m
        # 1.0
        idivf = params["idivf"].m_as(off_unit.dimensionless)
        if idivf == 0:
            raise RuntimeError("Found an idivf of 0.")
        torsion_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _process_rb_torsion_forces(openff_sys, openmm_sys):
    """
    Process Ryckaert-Bellemans torsions.
    """
    rb_force = openmm.RBTorsionForce()
    openmm_sys.addForce(rb_force)

    rb_torsion_handler = openff_sys["RBTorsions"]

    for top_key, pot_key in rb_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = rb_torsion_handler.potentials[pot_key].parameters

        c0 = params["C0"].m_as(off_unit.kilojoule / off_unit.mol)
        c1 = params["C1"].m_as(off_unit.kilojoule / off_unit.mol)
        c2 = params["C2"].m_as(off_unit.kilojoule / off_unit.mol)
        c3 = params["C3"].m_as(off_unit.kilojoule / off_unit.mol)
        c4 = params["C4"].m_as(off_unit.kilojoule / off_unit.mol)
        c5 = params["C5"].m_as(off_unit.kilojoule / off_unit.mol)

        rb_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        )


def _process_improper_torsion_forces(openff_sys, openmm_sys):
    """
    Process the Impropers section of an Interchange object.
    """
    if "ImproperTorsions" not in openff_sys.handlers.keys():
        return

    for force in openmm_sys.getForces():
        if type(force) == openmm.PeriodicTorsionForce:
            torsion_force = force
            break
    else:
        torsion_force = openmm.PeriodicTorsionForce()

    improper_torsion_handler = openff_sys["ImproperTorsions"]

    for top_key, pot_key in improper_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = improper_torsion_handler.potentials[pot_key].parameters

        k = params["k"].m_as(off_unit.kilojoule / off_unit.mol)
        periodicity = int(params["periodicity"])
        phase = params["phase"].m_as(off_unit.radian)
        idivf = int(params["idivf"])

        torsion_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _is_constrained(constrained_pairs: List[Tuple[int]], pair: Tuple[int, int]) -> bool:
    if (pair[0], pair[1]) in constrained_pairs:
        return True
    if (pair[1], pair[0]) in constrained_pairs:
        return True
    return False
