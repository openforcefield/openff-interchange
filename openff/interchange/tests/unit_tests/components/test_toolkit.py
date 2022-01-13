import pytest
from openff.toolkit.topology.molecule import Molecule

from openff.interchange.components.toolkit import _get_14_pairs


@pytest.mark.parametrize(
    ("smiles", "num_pairs"), [("C#C", 1), ("CCO", 12), ("C1=CC=CC=C1", 24)]
)
def test_get_14_pairs(smiles, num_pairs):
    mol = Molecule.from_smiles(smiles)
    assert len([*_get_14_pairs(mol)]) == num_pairs
    assert len([*_get_14_pairs(mol.to_topology())]) == num_pairs
