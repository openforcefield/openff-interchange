"""Pytest configuration."""
import pytest
from openff.toolkit import ForceField, Molecule

from openff.interchange._tests import get_test_file_path


@pytest.fixture()
def _simple_force_field():
    # TODO: Create a minimal force field for faster tests
    pass


@pytest.fixture()
def tip3p() -> ForceField:
    return ForceField("tip3p.offxml")


@pytest.fixture()
def tip4p() -> ForceField:
    return ForceField("tip4p_fb.offxml")


@pytest.fixture()
def gbsa_force_field() -> ForceField:
    return ForceField(
        "openff-2.0.0.offxml",
        get_test_file_path("gbsa.offxml"),
    )


@pytest.fixture(scope="session")
def water() -> Molecule:
    molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    molecule.generate_conformers(n_conformers=1)
    return molecule
