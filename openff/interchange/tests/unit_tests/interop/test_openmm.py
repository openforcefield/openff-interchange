import pytest
from openff.toolkit import ForceField, Molecule
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)

from openff.interchange import Interchange


class TestOpenMM:
    def test_no_nonbonded_force(self):
        """
        Ensure a SMIRNOFF-style force field can be exported to OpenMM even if no nonbonded handlers are present. For
        context, see https://github.com/openforcefield/openff-toolkit/issues/1102
        """

        sage = ForceField("openff_unconstrained-2.0.0.offxml")
        del sage._parameter_handlers["ToolkitAM1BCC"]
        del sage._parameter_handlers["LibraryCharges"]
        del sage._parameter_handlers["Electrostatics"]
        del sage._parameter_handlers["vdW"]

        water = Molecule.from_smiles("C")
        openmm_system = Interchange.from_smirnoff(sage, [water]).to_openmm()

        for force in openmm_system.getForces():
            if isinstance(force, NonbondedForce):
                pytest.fail("A NonbondedForce was found in the OpenMM system.")
            elif isinstance(force, PeriodicTorsionForce):
                assert force.getNumTorsions() == 0
            elif isinstance(force, HarmonicBondForce):
                assert force.getNumBonds() == 4
            elif isinstance(force, HarmonicAngleForce):
                assert force.getNumAngles() == 6
            else:
                pytest.fail(f"Unexpected force found, type: {type(force)}")
