import numpy as np
from simtk import openmm, unit

from openff.system.components.system import System
from openff.system.tests.energy_tests.report import EnergyReport

kj_mol = unit.kilojoule_per_mole


def get_openmm_energies(
    off_sys: System,
    round_positions=None,
    hard_cutoff: bool = True,
    electrostatics: bool = True,
) -> EnergyReport:

    omm_sys: openmm.System = off_sys.to_openmm()

    return _get_openmm_energies(
        omm_sys=omm_sys,
        box_vectors=off_sys.box,
        positions=off_sys.positions,
        round_positions=round_positions,
        hard_cutoff=hard_cutoff,
        electrostatics=electrostatics,
    )


def set_nonbonded_method(
    omm_sys: openmm.System,
    key: str,
    electrostatics: bool = True,
) -> openmm.System:

    if key == "cutoff":
        for force in omm_sys.getForces():
            if type(force) == openmm.NonbondedForce:
                force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
                force.setCutoffDistance(0.9 * unit.nanometer)
                force.setReactionFieldDielectric(1.0)
                force.setUseDispersionCorrection(False)
                force.setUseSwitchingFunction(False)
                if not electrostatics:
                    for i in range(force.getNumParticles()):
                        params = force.getParticleParameters(i)
                        force.setParticleParameters(
                            i,
                            0,
                            params[1],
                            params[2],
                        )

    elif key == "PME":
        for force in omm_sys.getForces():
            if type(force) == openmm.NonbondedForce:
                force.setNonbondedMethod(openmm.NonbondedForce.PME)
                force.setEwaldErrorTolerance(1e-6)

    return omm_sys


def _get_openmm_energies(
    omm_sys: openmm.System,
    box_vectors,
    positions,
    round_positions=None,
    hard_cutoff=False,
    electrostatics: bool = True,
) -> EnergyReport:

    if hard_cutoff:
        omm_sys = set_nonbonded_method(omm_sys, "cutoff", electrostatics=electrostatics)
    else:
        omm_sys = set_nonbonded_method(omm_sys, "PME")

    force_names = {force.__class__.__name__ for force in omm_sys.getForces()}
    group_to_force = {i: force_name for i, force_name in enumerate(force_names)}
    force_to_group = {force_name: i for i, force_name in group_to_force.items()}

    for force in omm_sys.getForces():
        force_name = force.__class__.__name__
        force.setForceGroup(force_to_group[force_name])

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(omm_sys, integrator)

    box_vectors = box_vectors.magnitude * unit.nanometer
    context.setPeriodicBoxVectors(*box_vectors)

    positions = positions.magnitude * unit.nanometer

    if round_positions is not None:
        rounded = np.round(positions, round_positions)
        context.setPositions(rounded)
    else:
        context.setPositions(positions)

    force_groups = {force.getForceGroup() for force in context.getSystem().getForces()}

    omm_energies = dict()

    for force_group in force_groups:
        state = context.getState(getEnergy=True, groups={force_group})
        omm_energies[group_to_force[force_group]] = state.getPotentialEnergy()
        del state

    # Fill in missing keys if system does not have all typical forces
    for required_key in [
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
    ]:
        if required_key not in omm_energies:
            omm_energies[required_key] = 0.0 * kj_mol

    del context
    del integrator

    report = EnergyReport()

    report.energies.update(
        {
            "Bond": omm_energies["HarmonicBondForce"],
            "Angle": omm_energies["HarmonicAngleForce"],
            "Torsion": omm_energies["PeriodicTorsionForce"],
            "Nonbonded": omm_energies["NonbondedForce"],
        }
    )

    if "CustomNonbondedForce" in omm_energies:
        report.energies["Nonbonded"] += omm_energies["CustomNonbondedForce"]

    return report
