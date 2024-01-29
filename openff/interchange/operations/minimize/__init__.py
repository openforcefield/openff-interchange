"""Engines for energy minimization."""

from openff.units import Quantity, unit

_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE = Quantity(
    10.0,
    unit.kilojoule_per_mole / unit.nanometer,
)
