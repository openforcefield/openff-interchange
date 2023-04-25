import parmed
import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.tests.utils import requires_openeye

from openff.interchange import Interchange
from openff.interchange._tests import get_test_file_path


@requires_openeye
@pytest.mark.parametrize(
    "molecule",
    [
        Molecule.from_file(get_test_file_path("lig_CHEMBL3265016-1.pdb")),
        Molecule.from_smiles("c1ccc2ccccc2c1"),
    ],
)
def test_atom_names_with_padding(molecule):
    # Unclear if the toolkit will always load PDBs with padded whitespace in name
    Interchange.from_smirnoff(
        ForceField("openff-2.0.0.offxml"),
        molecule.to_topology(),
    ).to_prmtop("tmp.prmtop")

    # Loading with ParmEd striggers #679 if exclusions lists are wrong
    parmed.load_file("tmp.prmtop")
