"""Engines for energy minimization."""

from openff.toolkit import Quantity

_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE = Quantity(10.0, "kilojoule_per_mole / nanometer")
