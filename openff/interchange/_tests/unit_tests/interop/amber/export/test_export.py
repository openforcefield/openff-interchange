import parmed
import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.tests.utils import requires_openeye

from openff.interchange import Interchange
from openff.interchange._tests import get_test_file_path


# pytest processes fixtures before the decorator can be applied
@requires_openeye
@pytest.mark.parametrize(
    "molecule",
    [
        "lig_CHEMBL3265016-1.pdb",
        "c1ccc2ccccc2c1",
    ],
)
def test_atom_names_with_padding(molecule):
    if molecule.endswith(".pdb"):
        molecule = Molecule(get_test_file_path(molecule).as_posix())
    else:
        molecule = Molecule.from_smiles(molecule)

    # Unclear if the toolkit will always load PDBs with padded whitespace in name
    Interchange.from_smirnoff(
        ForceField("openff-2.0.0.offxml"),
        molecule.to_topology(),
    ).to_prmtop("tmp.prmtop")

    # Loading with ParmEd striggers #679 if exclusions lists are wrong
    parmed.load_file("tmp.prmtop")
