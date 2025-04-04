import pytest
from openff.toolkit import Quantity, unit

from openff.interchange._tests import MoleculeWithConformer, needs_lmp
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.lammps import _process, get_lammps_energies


class TestProcess:
    @pytest.fixture
    def dummy_energies(self):
        energies = [Quantity(val, kj_mol) for val in [2.0, 1.0, 1.5, 0.5, -100.0, 0.1, -4, -400]]

        return {
            "Bond": energies[0],
            "Angle": energies[1],
            "ProperTorsion": energies[2],
            "ImproperTorsion": energies[3],
            "vdW": energies[4],
            "DispersionCorrection": energies[5],
            "ElectrostaticsShort": energies[6],
            "ElectrostaticsLong": energies[7],
        }

    def test_simple(self, dummy_energies):
        processed = _process(dummy_energies)

        for key in ["Bond", "Angle"]:
            assert processed[key] == dummy_energies[key]

        assert processed["Torsion"] == (dummy_energies["ProperTorsion"] + dummy_energies["ImproperTorsion"])
        assert processed["vdW"] == (dummy_energies["vdW"] + dummy_energies["DispersionCorrection"])
        assert processed["Electrostatics"] == (
            dummy_energies["ElectrostaticsShort"] + dummy_energies["ElectrostaticsLong"]
        )

    @pytest.mark.skip("Not implemented yet")
    def test_detailed(self, dummy_energies):
        assert "ImproperTorsion" in _process(
            dummy_energies,
            detailed=True,
        )


class TestLAMMPSDriver:
    @needs_lmp
    def test_can_write_without_torsions(self, sage):
        topology = MoleculeWithConformer.from_smiles("N").to_topology()
        topology.box_vectors = Quantity([4, 4, 4], unit.nanometer)

        assert get_lammps_energies(sage.create_interchange(topology))["Bond"].m == 0.0
