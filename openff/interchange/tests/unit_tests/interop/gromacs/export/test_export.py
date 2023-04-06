from openff.toolkit import Molecule

from openff.interchange import Interchange
from openff.interchange.tests import _BaseTest


class TestToGro(_BaseTest):
    def test_residue_names(self, sage):
        """Reproduce issue #642."""
        # This could maybe just test the behavior of _convert?
        ligand = Molecule.from_smiles("CCO")
        ligand.generate_conformers(n_conformers=1)

        for atom in ligand.atoms:
            atom.metadata["residue_name"] = "LIG"

        Interchange.from_smirnoff(
            sage,
            [ligand],
        ).to_gro("should_have_residue_names.gro")

        for line in open("should_have_residue_names.gro").readlines()[2:-2]:
            assert line[5:10] == "LIG  "
