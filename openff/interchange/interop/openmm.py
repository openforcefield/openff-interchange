"""Interfaces with OpenMM."""
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import openmm
from openff.toolkit.topology import Topology
from openff.units import unit as off_unit
from openff.units.openmm import from_openmm as from_openmm_unit
from openff.units.openmm import to_openmm as to_openmm_unit
from openmm import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import (
    UnimplementedCutoffMethodError,
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.interchange.interop.parmed import _lj_params_from_potential
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey

if TYPE_CHECKING:
    from openff.interchange import Interchange

kcal_mol = unit.kilocalorie_per_mole

kcal_ang = kcal_mol / unit.angstrom**2
kcal_rad = kcal_mol / unit.radian**2

kj_mol = unit.kilojoule_per_mole
kj_nm = kj_mol / unit.nanometer**2
kj_rad = kj_mol / unit.radian**2


def to_openmm(
    openff_sys,
    combine_nonbonded_forces: bool = False,
    add_constrained_forces: bool = False,
) -> openmm.System:
    """
    Convert an Interchange to an OpenmM System.

    Parameters
    ----------
    openff_sys : openff.interchange.Interchange
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
    openmm_sys = openmm.System()

    # OpenFF box stored implicitly as nm, and that happens to be what
    # OpenMM casts box vectors to if provided only an np.ndarray
    if openff_sys.box is not None:
        box = openff_sys.box.m_as(off_unit.nanometer)
        openmm_sys.setDefaultPeriodicBoxVectors(*box)

    # Add particles with appropriate masses
    # TODO: When to add virtual particles?
    for atom in openff_sys.topology.atoms:
        # Skip unit check for speed, toolkit should report mass in Dalton
        openmm_sys.addParticle(atom.mass.m)

    _process_nonbonded_forces(
        openff_sys, openmm_sys, combine_nonbonded_forces=combine_nonbonded_forces
    )
    constrained_pairs = _process_constraints(openff_sys, openmm_sys)
    _process_torsion_forces(openff_sys, openmm_sys)
    _process_improper_torsion_forces(openff_sys, openmm_sys)
    _process_angle_forces(
        openff_sys,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
    )
    _process_bond_forces(
        openff_sys,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
    )
    _process_virtual_sites(openff_sys, openmm_sys)

    return openmm_sys


def _process_constraints(openff_sys, openmm_sys):
    """
    Process the Constraints section of an Interchange object.
    """
    try:
        constraint_handler = openff_sys.handlers["Constraints"]
    except KeyError:
        return

    constrained_pairs = list()

    for top_key, pot_key in constraint_handler.slot_map.items():
        indices = top_key.atom_indices
        params = constraint_handler.constraints[pot_key].parameters
        distance = params["distance"]
        distance_omm = distance.m_as(off_unit.nanometer)

        constrained_pairs.append(tuple(sorted(indices)))
        openmm_sys.addConstraint(indices[0], indices[1], distance_omm)

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
        harmonic_angle_force.addPerAngleParameter("k")
        harmonic_angle_force.addPerAngleParameter("angle")
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
            k = params["k"].m_as(off_unit.kilojoule / off_unit.rad / off_unit.mol)
            angle = params["angle"].m_as(off_unit.radian)

            harmonic_angle_force.addAngle(
                indices[0],
                indices[1],
                indices[2],
                [k, angle],
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

    proper_torsion_handler = openff_sys.handlers["ProperTorsions"]

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

    rb_torsion_handler = openff_sys.handlers["RBTorsions"]

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

    improper_torsion_handler = openff_sys.handlers["ImproperTorsions"]

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


def _process_nonbonded_forces(openff_sys, openmm_sys, combine_nonbonded_forces=False):
    """
    Process the non-bonded handlers in an Interchange into corresponding openmm objects.

    This typically involves processing the vdW and Electrostatics sections of an Interchange object
    into a corresponding openmm.NonbondedForce (if `combine_nonbonded_forces=True`) or a
    collection of other forces (NonbondedForce, CustomNonbondedForce, CustomBondForce) if
    `combine_nonbondoed_forces=False`.

    """
    if "vdW" in openff_sys.handlers:
        vdw_handler = openff_sys.handlers["vdW"]

        vdw_cutoff = vdw_handler.cutoff.m_as(off_unit.angstrom) * unit.angstrom
        vdw_method = vdw_handler.method.lower()

        electrostatics_handler = openff_sys.handlers["Electrostatics"]
        electrostatics_method = electrostatics_handler.method.lower()

        if combine_nonbonded_forces:
            if vdw_handler.mixing_rule != "lorentz-berthelot":
                raise UnsupportedExportError(
                    "OpenMM's default NonbondedForce only supports Lorentz-Berthelot mixing rules."
                    "Try setting `combine_nonbonded_forces=False`."
                )

            non_bonded_force = openmm.NonbondedForce()
            openmm_sys.addForce(non_bonded_force)

            for _ in openff_sys.topology.atoms:
                non_bonded_force.addParticle(0.0, 1.0, 0.0)

            if vdw_method == "cutoff" and electrostatics_method == "pme":
                if openff_sys.box is None:
                    raise UnsupportedCutoffMethodError(
                        f"Combination of non-bonded cutoff methods {vdw_cutoff} (vdW) and "
                        f"{electrostatics_method} (Electrostatics) not currently supported with "
                        f"`combine_nonbonded_forces={combine_nonbonded_forces}` and "
                        f"`.box={openff_sys.box}`"
                    )
                else:
                    non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
                    non_bonded_force.setUseDispersionCorrection(True)
                    non_bonded_force.setCutoffDistance(vdw_cutoff)
                    non_bonded_force.setEwaldErrorTolerance(1.0e-4)

            elif vdw_method == "pme" and electrostatics_method == "pme":
                if openff_sys.box is None:
                    raise UnsupportedCutoffMethodError(
                        f"Combination of non-bonded cutoff methods {vdw_cutoff} (vdW) and "
                        f"{electrostatics_method} (Electrostatics) not valid with a non-periodic system "
                        f"(`.box={openff_sys.box}`)"
                    )
                else:
                    non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.LJPME)
                    non_bonded_force.setEwaldErrorTolerance(1.0e-4)
            else:
                raise UnimplementedCutoffMethodError(
                    f"Combination of non-bonded cutoff methods {vdw_cutoff} (vdW) and "
                    f"{electrostatics_method} (Electrostatics) not currently supported with "
                    f"`combine_nonbonded_forces={combine_nonbonded_forces}"
                )

        else:
            vdw_expression = vdw_handler.expression
            vdw_expression = vdw_expression.replace("**", "^")

            mixing_rule_expression = (
                "sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2); "
            )

            vdw_force = openmm.CustomNonbondedForce(
                vdw_expression + "; " + mixing_rule_expression
            )
            openmm_sys.addForce(vdw_force)
            vdw_force.addPerParticleParameter("sigma")
            vdw_force.addPerParticleParameter("epsilon")

            # TODO: Add virtual particles
            for _ in openff_sys.topology.atoms:
                vdw_force.addParticle([1.0, 0.0])

            if vdw_method == "cutoff":
                if openff_sys.box is None:
                    vdw_force.setNonbondedMethod(
                        openmm.NonbondedForce.CutoffNonPeriodic
                    )
                else:
                    vdw_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
                vdw_force.setUseLongRangeCorrection(True)
                vdw_force.setCutoffDistance(vdw_cutoff)

                _apply_switching_function(vdw_handler, vdw_force)

            elif vdw_method == "pme":
                if openff_sys.box is None:
                    raise UnsupportedCutoffMethodError(
                        "vdW method pme/ljpme is not valid for non-periodic systems."
                    )
                else:
                    raise UnsupportedCutoffMethodError(
                        "LJ-PME with split non-bonded forces is not supported due to openmm.CustomNonbondedForce "
                        "not supporting PME. If also using PME electrostatics, try `combine_nonbonded_forces=True`,  "
                        "which should produce a single force with NonbondedForce.LJPME, which uses PME for both "
                        "electrostatics and LJ forces tersm. If your use case would benenfit from split non-bonded "
                        "forces with LJPME, please file an feature request."
                    )

            electrostatics_force = openmm.NonbondedForce()
            openmm_sys.addForce(electrostatics_force)

            for _ in openff_sys.topology.atoms:
                electrostatics_force.addParticle(0.0, 1.0, 0.0)

            if electrostatics_method == "reaction-field":
                if openff_sys.box is None:
                    # TODO: Should this state be prevented from happening?
                    raise UnsupportedCutoffMethodError(
                        f"Electrostatics method {electrostatics_method} is not valid for a non-periodic interchange."
                    )
                else:
                    raise UnimplementedCutoffMethodError(
                        f"Electrostatics method {electrostatics_method} is not yet implemented."
                    )
            elif electrostatics_method == "pme":
                electrostatics_force.setNonbondedMethod(openmm.NonbondedForce.PME)
                electrostatics_force.setEwaldErrorTolerance(1.0e-4)
                electrostatics_force.setUseDispersionCorrection(True)
            elif electrostatics_method == "cutoff":
                raise UnsupportedCutoffMethodError(
                    "OpenMM does not clearly support cut-off electrostatics with no reaction-field attenuation."
                )
            else:
                raise UnsupportedCutoffMethodError(
                    f"Electrostatics method {electrostatics_method} not supported"
                )

        try:
            partial_charges = electrostatics_handler.charges_with_virtual_sites
        except AttributeError:
            partial_charges = electrostatics_handler.charges

        for top_key, pot_key in vdw_handler.slot_map.items():
            # TODO: Actually process virtual site vdW parameters here
            if type(top_key) != TopologyKey:
                continue
            atom_idx = top_key.atom_indices[0]

            partial_charge = partial_charges[top_key]
            # partial_charge = partial_charge.m_as(off_unit.elementary_charge)
            vdw_potential = vdw_handler.potentials[pot_key]
            # these are floats, implicitly angstrom and kcal/mol
            sigma, epsilon = _lj_params_from_potential(vdw_potential)
            sigma = sigma.m_as(off_unit.nanometer)
            epsilon = epsilon.m_as(off_unit.kilojoule / off_unit.mol)

            if combine_nonbonded_forces:
                non_bonded_force.setParticleParameters(
                    atom_idx,
                    partial_charge.m_as(off_unit.e),
                    sigma,
                    epsilon,
                )
            else:
                vdw_force.setParticleParameters(atom_idx, [sigma, epsilon])
                electrostatics_force.setParticleParameters(
                    atom_idx, partial_charge.m_as(off_unit.e), 0.0, 0.0
                )

    elif "Buckingham-6" in openff_sys.handlers:
        buck_handler = openff_sys.handlers["Buckingham-6"]

        non_bonded_force = openmm.CustomNonbondedForce(
            "A * exp(-B * r) - C * r ^ -6; A = sqrt(A1 * A2); B = 2 / (1 / B1 + 1 / B2); C = sqrt(C1 * C2)"
        )
        non_bonded_force.addPerParticleParameter("A")
        non_bonded_force.addPerParticleParameter("B")
        non_bonded_force.addPerParticleParameter("C")
        openmm_sys.addForce(non_bonded_force)

        for _ in openff_sys.topology.atoms:
            non_bonded_force.addParticle([0.0, 0.0, 0.0])

        if openff_sys.box is None:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            non_bonded_force.setCutoffDistance(buck_handler.cutoff * unit.angstrom)

        for top_key, pot_key in buck_handler.slot_map.items():
            atom_idx = top_key.atom_indices[0]

            # TODO: Add electrostatics
            params = buck_handler.potentials[pot_key].parameters
            a = to_openmm_unit(params["A"])
            b = to_openmm_unit(params["B"])
            c = to_openmm_unit(params["C"])
            non_bonded_force.setParticleParameters(atom_idx, [a, b, c])

        return

    if not combine_nonbonded_forces:
        # Attempting to match the value used internally by OpenMM; The source of this value is likely
        # https://github.com/openmm/openmm/issues/1149#issuecomment-250299854
        # 1 / * (4pi * eps0) * elementary_charge ** 2 / nanometer ** 2
        coul_const = 138.935456  # kJ/nm

        vdw_14_force = openmm.CustomBondForce("4*epsilon*((sigma/r)^12-(sigma/r)^6)")
        vdw_14_force.addPerBondParameter("sigma")
        vdw_14_force.addPerBondParameter("epsilon")
        vdw_14_force.setUsesPeriodicBoundaryConditions(True)
        coul_14_force = openmm.CustomBondForce(f"{coul_const}*qq/r")
        coul_14_force.addPerBondParameter("qq")
        coul_14_force.setUsesPeriodicBoundaryConditions(True)

        openmm_sys.addForce(vdw_14_force)
        openmm_sys.addForce(coul_14_force)

    # Need to create 1-4 exceptions, just to have a baseline for splitting out/modifying
    # It might be simpler to iterate over 1-4 pairs directly
    bonds = [
        sorted(openff_sys.topology.atom_index(a) for a in bond.atoms)
        for bond in openff_sys.topology.bonds
    ]

    if combine_nonbonded_forces:
        non_bonded_force.createExceptionsFromBonds(
            bonds=bonds,
            coulomb14Scale=electrostatics_handler.scale_14,
            lj14Scale=vdw_handler.scale_14,
        )
    else:
        electrostatics_force.createExceptionsFromBonds(
            bonds=bonds,
            coulomb14Scale=electrostatics_handler.scale_14,
            lj14Scale=vdw_handler.scale_14,
        )

        for i in range(electrostatics_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = electrostatics_force.getExceptionParameters(i)

            # If the interactions are both zero, assume this is a 1-2 or 1-3 interaction
            if q._value == 0 and eps._value == 0:
                pass
            else:
                # Assume this is a 1-4 interaction
                # Look up the vdW parameters for each particle
                sig1, eps1 = vdw_force.getParticleParameters(p1)
                sig2, eps2 = vdw_force.getParticleParameters(p2)
                q1, _, _ = electrostatics_force.getParticleParameters(p1)
                q2, _, _ = electrostatics_force.getParticleParameters(p2)

                # manually compute and set the 1-4 interactions
                sig_14 = (sig1 + sig2) * 0.5
                eps_14 = (eps1 * eps2) ** 0.5 * vdw_handler.scale_14
                qq = q1 * q2 * electrostatics_handler.scale_14

                vdw_14_force.addBond(p1, p2, [sig_14, eps_14])
                coul_14_force.addBond(p1, p2, [qq])
            vdw_force.addExclusion(p1, p2)
            # electrostatics_force.addExclusion(p1, p2)
            electrostatics_force.setExceptionParameters(i, p1, p2, 0.0, 0.0, 0.0)
            # vdw_force.setExceptionParameters(i, p1, p2, 0.0, 0.0, 0.0)


def _process_virtual_sites(openff_sys, openmm_sys):
    try:
        virtual_site_handler = openff_sys.handlers["VirtualSites"]
    except KeyError:
        return

    vdw_handler = openff_sys.handlers["vdW"]
    coul_handler = openff_sys.handlers["Electrostatics"]

    # TODO: Handle case of split-out non-bonded forces
    non_bonded_force = [
        f for f in openmm_sys.getForces() if type(f) == openmm.NonbondedForce
    ][0]

    for virtual_site_key in virtual_site_handler.slot_map:
        vdw_key = vdw_handler.slot_map.get(virtual_site_key)
        coul_key = coul_handler.slot_map.get(virtual_site_key)
        if vdw_key is None and coul_key is None:
            raise Exception(
                f"Virtual site {virtual_site_key} is not associated with any "
                "vdW or electrostatics interactions"
            )

        if coul_key is None:
            charge = 0.0
        else:
            charge = coul_handler.charges_with_virtual_sites[virtual_site_key].m_as(
                off_unit.elementary_charge,
            )
        if vdw_key is None:
            sigma = 1.0
            epsilon = 0.0
        else:
            vdw_parameters = vdw_handler.potentials[vdw_key].parameters
            sigma = vdw_parameters["sigma"].m_as(
                off_unit.nanometer,
            )
            epsilon = vdw_parameters["epsilon"].m_as(
                off_unit.Unit(str(kj_mol)),
            )

        virtual_site_index = openmm_sys.addParticle(mass=0.0)

        openmm_virtual_site = _create_virtual_site(virtual_site_key, openff_sys)

        openmm_sys.setVirtualSite(virtual_site_index, openmm_virtual_site)

        non_bonded_force.addParticle(charge, sigma, epsilon)

        for parent_atom_index in virtual_site_key.atom_indices:
            non_bonded_force.addException(
                parent_atom_index, virtual_site_index, 0.0, 0.0, 0.0, replace=True
            )


def _create_virtual_site(
    virtual_site_key: "VirtualSiteKey",
    interchange: "Interchange",
) -> "openmm.LocalCoordinatesSites":

    parent_atoms = virtual_site_key.atom_indices
    origin_weight, x_direction, y_direction = interchange[
        "VirtualSites"
    ]._get_local_frame_weights(virtual_site_key)
    displacement = interchange["VirtualSites"]._get_local_frame_position(
        virtual_site_key
    )

    x, y, z = ((v / v.units).m for v in displacement)
    # x, y, z = displacement / displacement.units

    parent_atom_positions = []
    for parent_atom in parent_atoms:
        parent_atom_positions.append(interchange.positions[parent_atom])

    _origin_weight = np.atleast_2d(origin_weight)
    parent_atom_positions = np.atleast_2d(parent_atom_positions)

    origin = np.dot(_origin_weight, parent_atom_positions).sum(axis=0)

    x_axis, y_axis = np.dot(
        np.vstack((x_direction, y_direction)), parent_atom_positions
    )

    z_axis = np.cross(x_axis, y_axis)
    y_axis = np.cross(z_axis, x_axis)

    def _normalize(axis):
        l = np.linalg.norm(axis)  # noqa
        if l > 0.0:
            axis /= l
        return axis

    x_axis, y_axis, z_axis = map(_normalize, (x_axis, y_axis, z_axis))

    position = origin + x * x_axis + y * y_axis + z * z_axis

    return openmm.LocalCoordinatesSite(
        parent_atoms,
        origin_weight,
        x_direction,
        y_direction,
        position,
    )


def _apply_switching_function(vdw_handler, force: openmm.NonbondedForce):
    if not hasattr(force, "setUseSwitchingFunction"):
        raise ValueError(
            "Attempting to set switching funcntion on an OpenMM force that does nont support it."
            f"Passed force of type {type(force)}."
        )
    if getattr(vdw_handler, "switch_width", None) is None:
        force.setUseSwitchingFunction(False)
    elif vdw_handler.switch_width.m == 0.0:
        force.setUseSwitchingFunction(False)
    else:
        switching_distance = (vdw_handler.cutoff - vdw_handler.switch_width).m_as(
            off_unit.angstrom
        )

        if switching_distance < 0:
            raise UnsupportedCutoffMethodError(
                "Found a `switch_width` greater than the cutoff distance. It's not clear "
                "what this means and it's probably invalid. Found "
                f"switch_width{vdw_handler.switch_width} and cutoff {vdw_handler.cutoff}"
            )

        switching_distance = unit.Quantity(switching_distance, unit.angstrom)

        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(switching_distance)


def from_openmm(topology=None, system=None, positions=None, box_vectors=None):
    """Create an Interchange object from OpenMM data."""
    from openff.interchange import Interchange

    openff_sys = Interchange()

    if system:
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                vdw, coul = _convert_nonbonded_force(force)
                openff_sys.add_handler(handler_name="vdW", handler=vdw)
                openff_sys.add_handler(handler_name="Electrostatics", handler=coul)
            if isinstance(force, openmm.HarmonicBondForce):
                bond_handler = _convert_harmonic_bond_force(force)
                openff_sys.add_handler(handler_name="Bonds", handler=bond_handler)
            if isinstance(force, openmm.HarmonicAngleForce):
                angle_handler = _convert_harmonic_angle_force(force)
                openff_sys.add_handler(handler_name="Angles", handler=angle_handler)
            if isinstance(force, openmm.PeriodicTorsionForce):
                proper_torsion_handler = _convert_periodic_torsion_force(force)
                openff_sys.add_handler(
                    handler_name="ProperTorsions",
                    handler=proper_torsion_handler,
                )

    if topology is not None:
        from openff.interchange.components.toolkit import _simple_topology_from_openmm

        openff_topology = _simple_topology_from_openmm(topology)

        openff_sys.topology = openff_topology

    if positions is not None:
        openff_sys.positions = positions

    if box_vectors is not None:
        openff_sys.box = box_vectors

    return openff_sys


def _convert_nonbonded_force(force):
    from openff.interchange.components.smirnoff import (
        SMIRNOFFElectrostaticsHandler,
        SMIRNOFFvdWHandler,
    )

    vdw_handler = SMIRNOFFvdWHandler()
    electrostatics = SMIRNOFFElectrostaticsHandler(scale_14=0.833333, method="pme")

    n_parametrized_particles = force.getNumParticles()

    for idx in range(n_parametrized_particles):
        charge, sigma, epsilon = force.getParticleParameters(idx)
        top_key = TopologyKey(atom_indices=(idx,))
        pot_key = PotentialKey(id=f"{idx}")
        pot = Potential(
            parameters={
                "sigma": from_openmm_unit(sigma),
                "epsilon": from_openmm_unit(epsilon),
            }
        )
        vdw_handler.slot_map.update({top_key: pot_key})
        vdw_handler.potentials.update({pot_key: pot})

        electrostatics.slot_map.update({top_key: pot_key})
        electrostatics.potentials.update(
            {pot_key: Potential(parameters={"charge": from_openmm_unit(charge)})}
        )

    if force.getNonbondedMethod() == openmm.NonbondedForce.PME:
        electrostatics.method = "pme"
        vdw_handler.method = "cutoff"
    elif force.getNonbondedMethod() in {
        openmm.NonbondedForce.CutoffPeriodic,
        openmm.NonbondedForce.CutoffNonPeriodic,
    }:
        # TODO: Store reaction-field dielectric
        electrostatics.method = "reaction-field"
        vdw_handler.method = "cutoff"
    elif force.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
        electrostatics.method = "no-cutoff"
        vdw_handler.method = "no-cutoff"

    if vdw_handler.method == "cutoff":
        vdw_handler.cutoff = force.getCutoffDistance()
    electrostatics.cutoff = force.getCutoffDistance()

    return vdw_handler, electrostatics


def _convert_harmonic_bond_force(force):
    from openff.interchange.components.smirnoff import SMIRNOFFBondHandler

    bond_handler = SMIRNOFFBondHandler()

    n_parametrized_bonds = force.getNumBonds()

    for idx in range(n_parametrized_bonds):
        atom1, atom2, length, k = force.getBondParameters(idx)
        top_key = TopologyKey(atom_indices=(atom1, atom2))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}")
        pot = Potential(
            parameters={"length": from_openmm_unit(length), "k": from_openmm_unit(k)}
        )

        bond_handler.slot_map.update({top_key: pot_key})
        bond_handler.potentials.update({pot_key: pot})

    return bond_handler


def _convert_harmonic_angle_force(force):
    from openff.interchange.components.smirnoff import SMIRNOFFAngleHandler

    angle_handler = SMIRNOFFAngleHandler()

    n_parametrized_angles = force.getNumAngles()

    for idx in range(n_parametrized_angles):
        atom1, atom2, atom3, angle, k = force.getAngleParameters(idx)
        top_key = TopologyKey(atom_indices=(atom1, atom2, atom3))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}")
        pot = Potential(
            parameters={"angle": from_openmm_unit(angle), "k": from_openmm_unit(k)}
        )

        angle_handler.slot_map.update({top_key: pot_key})
        angle_handler.potentials.update({pot_key: pot})

    return angle_handler


def _convert_periodic_torsion_force(force):
    # TODO: Can impropers be separated out from a PeriodicTorsionForce?
    # Maybe by seeing if a quartet is in mol/top.propers or .impropers
    from openff.interchange.components.smirnoff import SMIRNOFFProperTorsionHandler

    proper_torsion_handler = SMIRNOFFProperTorsionHandler()

    n_parametrized_torsions = force.getNumTorsions()

    for idx in range(n_parametrized_torsions):
        atom1, atom2, atom3, atom4, per, phase, k = force.getTorsionParameters(idx)
        # TODO: Process layered torsions
        top_key = TopologyKey(atom_indices=(atom1, atom2, atom3, atom4), mult=0)
        while top_key in proper_torsion_handler.slot_map:
            top_key.mult: int = top_key.mult + 1

        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}-{atom4}", mult=top_key.mult)
        pot = Potential(
            parameters={
                "periodicity": int(per) * unit.dimensionless,
                "phase": from_openmm_unit(phase),
                "k": from_openmm_unit(k),
                "idivf": 1 * unit.dimensionless,
            }
        )

        proper_torsion_handler.slot_map.update({top_key: pot_key})
        proper_torsion_handler.potentials.update({pot_key: pot})

    return proper_torsion_handler


def _to_pdb(file_path: Union[Path, str], topology: Topology, positions):
    from openff.units.openmm import to_openmm
    from openmm import app

    openmm_topology = topology.to_openmm(ensure_unique_atom_names=False)

    positions = to_openmm(positions)

    with open(file_path, "w") as outfile:
        app.PDBFile.writeFile(openmm_topology, positions, outfile)


def get_nonbonded_force_from_openmm_system(omm_system):
    """Get a single NonbondedForce object with an OpenMM System."""
    for force in omm_system.getForces():
        if type(force) == openmm.NonbondedForce:
            return force


def get_partial_charges_from_openmm_system(omm_system):
    """Get partial charges from an OpenMM interchange as a unit.Quantity array."""
    # TODO: deal with virtual sites
    n_particles = omm_system.getNumParticles()
    force = get_nonbonded_force_from_openmm_system(omm_system)
    # TODO: don't assume the partial charge will always be parameter 0
    # partial_charges = [openmm_to_pint(force.getParticleParameters(idx)[0]) for idx in range(n_particles)]
    partial_charges = [
        force.getParticleParameters(idx)[0] / unit.elementary_charge
        for idx in range(n_particles)
    ]

    return partial_charges


def _is_constrained(constrained_pairs: List[Tuple[int]], pair: Tuple[int, int]) -> bool:
    if (pair[0], pair[1]) in constrained_pairs:
        return True
    if (pair[1], pair[0]) in constrained_pairs:
        return True
    return False
