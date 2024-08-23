"""Minimize energy using OpenMM."""

from typing import TYPE_CHECKING

from openff.toolkit import Quantity
from openff.utilities.utilities import requires_package

from openff.interchange.exceptions import MinimizationError, MissingPositionsError

if TYPE_CHECKING:
    from openff.interchange import Interchange


@requires_package("openmm")
def minimize_openmm(
    interchange: "Interchange",
    tolerance: Quantity,
    max_iterations: int,
) -> Quantity:
    """Minimize the energy of a system using OpenMM."""
    import openmm
    import openmm.unit
    from openff.units.openmm import from_openmm

    simulation = interchange.to_openmm_simulation(
        integrator=openmm.LangevinMiddleIntegrator(
            293.15 * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            2.0 * openmm.unit.femtosecond,
        ),
        combine_nonbonded_forces=False,
    )

    simulation.context.computeVirtualSites()

    try:
        simulation.minimizeEnergy(
            tolerance=tolerance.to_openmm(),
            maxIterations=max_iterations,
        )

    except openmm.OpenMMException as error:
        if "Particle positions have not been set" in str(error):
            raise MissingPositionsError(
                f"Cannot minimize without positions. Found {interchange.positions=}.",
            ) from error
        else:
            raise MinimizationError("OpenMM Minimization failed.") from error

    # Assume that all virtual sites are placed at the _end_, so the 0th through
    # (number of atoms)th positions are the massive particles
    return from_openmm(
        simulation.context.getState(
            getPositions=True,
        ).getPositions(
                    asNumpy=True,
        )[
            : interchange.positions.shape[0],  # type: ignore[union-attr]
            :,
        ],
    )
