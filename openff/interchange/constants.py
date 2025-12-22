"""
Commonly-used constants.
"""

from openff.units import Unit

kj_mol = Unit("kilojoule / mol")
kcal_mol = Unit("kilocalorie_per_mole")

kcal_ang = Unit("kilojoule / mol / angstrom**2")
kcal_rad = Unit("kilojoule / mol / radian**2")

kj_nm = Unit("kilojoule/ mol / nanometer**2")
kj_rad = Unit("kilojoule/ mol / radian**2")

kcal_mol_a2 = Unit("kilocalorie / mol / angstrom**2")
kcal_mol_rad2 = Unit("kilocalorie / mol / radian**2")

_PME = "Ewald3D-ConductingBoundary"
AMBER_COULOMBS_CONSTANT = 18.2223
