"""Minimize energy using OpenMM."""
from typing import TYPE_CHECKING

from openff.units import Quantity
from openff.utilities.utilities import requires_package

if TYPE_CHECKING:
    from openff.interchange import Interchange


@requires_package("openmm")
def minimize_openmm(
    interchange: "Interchange",
    tolerance: Quantity,
    max_iterations: int,
) -> Quantity:
    """Minimize the energy of a system using OpenMM."""
    import openmm.unit
    from openff.units.openmm import from_openmm

    simulation = interchange.to_openmm_simulation(
        openmm.LangevinMiddleIntegrator(
            293.15 * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            2.0 * openmm.unit.femtosecond,
        ),
    )

    simulation.minimizeEnergy(
        tolerance=tolerance.to_openmm(),
        maxIterations=max_iterations,
    )

    return from_openmm(simulation.context.getState(getPositions=True).getPositions())
