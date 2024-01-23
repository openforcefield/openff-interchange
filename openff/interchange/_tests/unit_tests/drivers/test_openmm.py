import pytest
from openff.toolkit import ForceField
from openff.utilities.utilities import has_package

from openff.interchange._tests import MoleculeWithConformer, get_test_file_path
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.openmm import _process, get_openmm_energies

if has_package("openmm"):
    import openmm


class TestProcess:
    pytest.importorskip("openmm")

    @pytest.fixture()
    def dummy_system(self):
        system = openmm.System()
        for force in [
            openmm.PeriodicTorsionForce,
            openmm.HarmonicAngleForce,
            openmm.HarmonicBondForce,
            openmm.RBTorsionForce,
            openmm.NonbondedForce,
        ]:
            system.addForce(force())

        return system

    @pytest.fixture()
    def dummy_system_split(self, dummy_system):
        dummy_system.addForce(openmm.CustomNonbondedForce("sigma*epsilon"))
        dummy_system.addForce(openmm.CustomBondForce("sigma*epsilon"))
        dummy_system.addForce(openmm.CustomBondForce("qq"))

        return dummy_system

    def test_simple(self, dummy_system):
        processed = _process(
            {
                0: 1.1 * kj_mol,
                1: 2.2 * kj_mol,
                2: 3.3 * kj_mol,
                3: 4.4 * kj_mol,
                4: 5.5 * kj_mol,
            },
            dummy_system,
            True,
            False,
        )

        assert processed["Bond"].m_as(kj_mol) == 3.3
        assert processed["Angle"].m_as(kj_mol) == 2.2
        assert processed["Torsion"].m_as(kj_mol) == 1.1
        assert processed["RBTorsion"].m_as(kj_mol) == 4.4
        assert processed["Nonbonded"].m_as(kj_mol) == 5.5

    def test_split_forces(self, dummy_system_split):
        processed = _process(
            {
                0: 1.1 * kj_mol,
                1: 2.2 * kj_mol,
                2: 3.3 * kj_mol,
                3: 4.4 * kj_mol,
                4: 0.5 * kj_mol,
                5: -1 * kj_mol,
                6: -2 * kj_mol,  # vdW
                7: -3 * kj_mol,  # Electrostatics
            },
            dummy_system_split,
            False,
            True,
        )

        assert processed["Electrostatics"].m_as(kj_mol) == 0.5
        assert processed["Electrostatics 1-4"].m_as(kj_mol) == -3
        assert processed["vdW"].m_as(kj_mol) == -1
        assert processed["vdW 1-4"].m_as(kj_mol) == -2


class TestReportWithPlugins:
    pytest.importorskip("smirnoff_plugins")

    @pytest.fixture()
    def ligand(self):
        return MoleculeWithConformer.from_smiles("CC[C@@](/C=C\\[H])(C=C)O")

    @pytest.fixture()
    def de_force_field(self) -> ForceField:
        return ForceField(
            get_test_file_path("de-force-1.0.1.offxml"),
            load_plugins=True,
        )

    @pytest.mark.parametrize("detailed", [True, False])
    def test_nonzero_vdw(self, ligand, de_force_field, detailed):
        energies = get_openmm_energies(
            de_force_field.create_interchange(ligand.to_topology()),
            combine_nonbonded_forces=False,
            detailed=detailed,
        )

        assert energies["vdW"].m != 0

        if detailed:
            assert energies["vdW 1-4"].m != 0

    def test_detailed_same_total_energy(self, ligand, de_force_field):
        assert get_openmm_energies(
            de_force_field.create_interchange(ligand.to_topology()),
            combine_nonbonded_forces=False,
            detailed=True,
        ).total_energy.m == pytest.approx(
            get_openmm_energies(
                de_force_field.create_interchange(ligand.to_topology()),
                combine_nonbonded_forces=False,
                detailed=False,
            ).total_energy.m,
        )
