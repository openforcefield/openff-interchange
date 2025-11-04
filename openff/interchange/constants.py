"""
Commonly-used constants.
"""

from openff.units import Unit

_PME = "Ewald3D-ConductingBoundary"
kj_mol = Unit("kilojoule / mol")
kcal_mol = Unit("kilocalorie_per_mole")

kcal_ang = kcal_mol / Unit("angstrom**2")
kcal_rad = kcal_mol / Unit("radian**2")

kj_nm = kj_mol / Unit("nanometer**2")
kj_rad = kj_mol / Unit("radian**2")

AMBER_COULOMBS_CONSTANT = 18.2223
kcal_mol_a2 = kcal_mol / Unit("angstrom**2")
kcal_mol_rad2 = kcal_mol / Unit("radian**2")
