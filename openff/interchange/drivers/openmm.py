"""Functions for running energy evluations with OpenMM."""
import warnings
from typing import Dict, Optional

import numpy
import openmm
import openmm.unit
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import CannotInferNonbondedEnergyError
from openff.interchange.interop.openmm._positions import to_openmm_positions


def get_openmm_energies(
    interchange: Interchange,
    round_positions: Optional[int] = None,
    combine_nonbonded_forces: bool = True,
    detailed: bool = False,
    platform: str = "Reference",
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by OpenMM.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    interchange : openff.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    round_positions : int, optional
        The number of decimal places, in nanometers, to round positions. This can be useful when
        comparing to i.e. GROMACS energies, in which positions may be rounded.
    combine_nonbonded_forces : bool, default=False
        Whether or not to combine all non-bonded interactions (vdW, short- and long-range
        ectrostaelectrostatics, and 1-4 interactions) into a single openmm.NonbondedForce.
    platform : str, default="Reference"
        The name of the platform (`openmm.Platform`) used by OpenMM in this calculation.
    detailed : bool, default=False
        Attempt to report energies with more granularity. Not guaranteed to be compatible with all values
        of other arguments. Useful for debugging.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    if "VirtualSites" in interchange.collections:
        if len(interchange["VirtualSites"].key_map) > 0:
            if not combine_nonbonded_forces:
                warnings.warn(
                    "Collecting energies from split forces with virtual sites is experimental",
                    UserWarning,
                    stacklevel=2,
                )

            has_virtual_sites = True
        else:
            has_virtual_sites = False
    else:
        has_virtual_sites = False

    system: openmm.System = interchange.to_openmm(
        combine_nonbonded_forces=combine_nonbonded_forces,
    )

    box_vectors: openmm.unit.Quantity = (
        None if interchange.box is None else interchange.box.to_openmm()
    )

    positions: openmm.unit.Quantity = to_openmm_positions(
        interchange,
        include_virtual_sites=has_virtual_sites,
    )

    return _process(
        _get_openmm_energies(
            system=system,
            box_vectors=box_vectors,
            positions=positions,
            round_positions=round_positions,
            platform=platform,
        ),
        combine_nonbonded_forces=combine_nonbonded_forces,
        detailed=detailed,
        system=system,
    )


def _get_openmm_energies(
    system: openmm.System,
    box_vectors: Optional[openmm.unit.Quantity],
    positions: openmm.unit.Quantity,
    round_positions: Optional[int],
    platform: str,
) -> Dict[int, unit.Quantity]:
    """Given prepared `openmm` objects, run a single-point energy calculation."""
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)

    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName(platform),
    )

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)

    context.setPositions(
        numpy.round(positions, round_positions)
        if round_positions is not None
        else positions,
    )

    raw_energies: Dict[int, openmm.Force] = dict()

    for index in range(system.getNumForces()):
        state = context.getState(getEnergy=True, groups={index})
        raw_energies[index] = state.getPotentialEnergy()
        del state

    del context
    del integrator

    return raw_energies


def _process(
    raw_energies: Dict[int, openmm.Force],
    system: openmm.System,
    combine_nonbonded_forces: bool,
    detailed: bool,
) -> EnergyReport:
    staged: Dict[str, unit.Quantity] = dict()

    valence_map = {
        openmm.HarmonicBondForce: "Bond",
        openmm.HarmonicAngleForce: "Angle",
        openmm.PeriodicTorsionForce: "Torsion",
    }

    n_rb = len(
        [
            force
            for force in system.getForces()
            if isinstance(force, openmm.RBTorsionForce)
        ],
    )
    n_periodic = len(
        [
            force
            for force in system.getForces()
            if isinstance(force, openmm.PeriodicTorsionForce)
        ],
    )

    if n_rb * n_periodic > 0:
        raise NotImplementedError(
            "Cannot process systems with both PeriodicTorsionForce and RBTorsionForce",
        )

    # This assumes that only custom forces will have duplicate instances
    for index, raw_energy in raw_energies.items():
        force = system.getForce(index)

        if type(force) in valence_map:
            staged[valence_map[type(force)]] = raw_energy

        elif type(force) in [
            openmm.NonbondedForce,
            openmm.CustomNonbondedForce,
            openmm.CustomBondForce,
        ]:
            if combine_nonbonded_forces:
                assert isinstance(force, openmm.NonbondedForce)

                staged["Nonbonded"] = raw_energy

                continue

            else:
                if isinstance(force, openmm.NonbondedForce):
                    staged["Electrostatics"] = raw_energy

                elif isinstance(force, openmm.CustomNonbondedForce):
                    staged["vdW"] = raw_energy

                elif isinstance(force, openmm.CustomBondForce):
                    print(force.getEnergyFunction())
                    if "qq" in force.getEnergyFunction():
                        staged["Electrostatics 1-4"] = raw_energy
                    else:
                        staged["vdW 1-4"] = raw_energy

                else:
                    raise CannotInferNonbondedEnergyError()

    if detailed:
        processed = staged

    else:
        processed = {
            key: staged[key] for key in ["Bond", "Angle", "Torsion"] if key in staged
        }

        nonbonded_energies = [
            staged[key]
            for key in [
                "Nonbonded",
                "Electrostatics",
                "vdW",
                "Electrostatics 1-4",
                "vdW 1-4",
            ]
            if key in staged
        ]

        # Array inference acts up if given a 1-list of Quantity
        processed["Nonbonded"] = (
            nonbonded_energies[0]
            if len(nonbonded_energies) == 1
            else numpy.sum(nonbonded_energies)
        )

    return EnergyReport(energies=processed)
