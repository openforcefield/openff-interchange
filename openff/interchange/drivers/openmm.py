from typing import Dict

import numpy as np
from simtk import openmm, unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport

kj_mol = unit.kilojoule_per_mole


def get_openmm_energies(
    off_sys: Interchange,
    round_positions=None,
    hard_cutoff: bool = False,
    electrostatics: bool = True,
    combine_nonbonded_forces: bool = False,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by OpenMM.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.interchange.components.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    round_positions : int, optional
        The number of decimal places, in nanometers, to round positions. This can be useful when
        comparing to i.e. GROMACS energies, in which positions may be rounded.
    writer : str, default="internal"
        A string key identifying the backend to be used to write OpenMM files. The
        default value of `"internal"` results in this package's exporters being used.
    hard_cutoff : bool, default=True
        Whether or not to apply a hard cutoff (no switching function or disperson correction)
        to the `openmm.NonbondedForce` in the generated `openmm.System`. Note that this will
        truncate electrostatics to the non-bonded cutoff.
    electrostatics : bool, default=True
        A boolean indicating whether or not electrostatics should be included in the energy
        calculation.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """

    positions = off_sys.positions

    if "VirtualSites" in off_sys.handlers:
        if len(off_sys["VirtualSites"].slot_map) > 0:
            if not combine_nonbonded_forces:
                raise NotImplementedError(
                    "Cannot yet split out NonbondedForce components while virtual sites are present."
                )

            n_virtual_sites = len(off_sys["VirtualSites"].slot_map)

            # TODO: Actually compute virtual site positions based on initial conformers
            virtual_site_positions = np.zeros((n_virtual_sites, 3))
            virtual_site_positions *= off_sys.positions.units
            positions = np.vstack([positions, virtual_site_positions])

    omm_sys: openmm.System = off_sys.to_openmm(
        combine_nonbonded_forces=combine_nonbonded_forces
    )

    return _get_openmm_energies(
        omm_sys=omm_sys,
        box_vectors=off_sys.box,
        positions=positions,
        round_positions=round_positions,
        hard_cutoff=hard_cutoff,
        electrostatics=electrostatics,
    )


def _get_openmm_energies(
    omm_sys: openmm.System,
    box_vectors,
    positions,
    round_positions=None,
    hard_cutoff=False,
    electrostatics: bool = True,
) -> EnergyReport:
    """Given a prepared `openmm.System`, run a single-point energy calculation."""
    """\
    if hard_cutoff:
        omm_sys = _set_nonbonded_method(
            omm_sys, "cutoff", electrostatics=electrostatics
        )
    else:
        omm_sys = _set_nonbonded_method(omm_sys, "PME")
    """

    for idx, force in enumerate(omm_sys.getForces()):
        force.setForceGroup(idx)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(omm_sys, integrator)

    if box_vectors is not None:
        if not isinstance(box_vectors, (unit.Quantity, list)):
            box_vectors = box_vectors.magnitude * unit.nanometer
        context.setPeriodicBoxVectors(*box_vectors)

    if isinstance(positions, unit.Quantity):
        # Convert list of Vec3 into a NumPy array
        positions = np.asarray(positions.value_in_unit(unit.nanometer)) * unit.nanometer
    else:
        positions = positions.magnitude * unit.nanometer

    if round_positions is not None:
        rounded = np.round(positions, round_positions)
        context.setPositions(rounded)
    else:
        context.setPositions(positions)

    raw_energies = dict()
    omm_energies = dict()

    for idx in range(omm_sys.getNumForces()):
        state = context.getState(getEnergy=True, groups={idx})
        raw_energies[idx] = state.getPotentialEnergy()
        del state

    # This assumes that only custom forces will have duplicate instances
    for key in raw_energies:
        force = omm_sys.getForce(key)
        if type(force) == openmm.HarmonicBondForce:
            omm_energies["HarmonicBondForce"] = raw_energies[key]
        elif type(force) == openmm.HarmonicAngleForce:
            omm_energies["HarmonicAngleForce"] = raw_energies[key]
        elif type(force) == openmm.PeriodicTorsionForce:
            omm_energies["PeriodicTorsionForce"] = raw_energies[key]
        elif type(force) in [
            openmm.NonbondedForce,
            openmm.CustomNonbondedForce,
            openmm.CustomBondForce,
        ]:
            if "Nonbonded" in omm_energies:
                omm_energies["Nonbonded"] += raw_energies[key]
            else:
                omm_energies["Nonbonded"] = raw_energies[key]

    # Fill in missing keys if interchange does not have all typical forces
    for required_key in [
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
    ]:
        if not any(required_key in val for val in omm_energies):
            pass  # omm_energies[required_key] = 0.0 * kj_mol

    del context
    del integrator

    report = EnergyReport()

    report.update_energies(
        {
            "Bond": omm_energies.get("HarmonicBondForce", 0.0 * kj_mol),
            "Angle": omm_energies.get("HarmonicAngleForce", 0.0 * kj_mol),
            "Torsion": _canonicalize_torsion_energies(omm_energies),
            "Nonbonded": omm_energies.get(
                "Nonbonded", _canonicalize_nonbonded_energies(omm_energies)
            ),
        }
    )

    report.energies.pop("vdW")
    report.energies.pop("Electrostatics")

    return report


def _canonicalize_nonbonded_energies(energies: Dict):
    omm_nonbonded = 0.0 * kj_mol
    for key in ["NonbondedForce", "CustomNonbondedForce", "CustomBondForce"]:
        try:
            omm_nonbonded += energies[key]
        except KeyError:
            pass

    return omm_nonbonded


def _canonicalize_torsion_energies(energies: Dict):
    omm_torsion = 0.0 * kj_mol
    for key in ["PeriodicTorsionForce", "RBTorsionForce"]:
        try:
            omm_torsion += energies[key]
        except KeyError:
            pass

    return omm_torsion
