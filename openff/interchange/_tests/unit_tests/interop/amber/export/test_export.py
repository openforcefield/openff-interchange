import parmed
import pytest
from openff.toolkit import ForceField, Molecule

from openff.interchange import Interchange
from openff.interchange._tests import get_test_file_path


def test_atom_names_with_padding():
    # Unclear if the toolkit will always load PDBs with padded whitespace in name
    Interchange.from_smirnoff(
        ForceField("openff-2.0.0.offxml"),
        Molecule.from_file(get_test_file_path("ethanol.pdb")).to_topology(),
    ).to_prmtop("tmp.prmtop")

    # #679 persists, but #678 would cause an earlier error
    with pytest.raises(parmed.topologyobjects.MoleculeError):
        parmed.load_file("tmp.prmtop")
