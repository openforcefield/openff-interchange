"""Test SMIRNOFF-GROMACS conversion."""
from openff.toolkit import Molecule

from openff.interchange import Interchange
from openff.interchange.smirnoff._gromacs import _convert
from openff.interchange.tests import _BaseTest


class TestConvert(_BaseTest):
    def test_residue_names(self, sage):
        """Reproduce issue #642."""
        ligand = Molecule.from_smiles("CCO")
        ligand.generate_conformers(n_conformers=1)

        for atom in ligand.atoms:
            atom.metadata["residue_name"] = "LIG"

        system = _convert(
            Interchange.from_smirnoff(
                sage,
                [ligand],
            ),
        )

        for molecule_type in system.molecule_types.values():
            for atom in molecule_type.atoms:
                assert atom.residue_name == "LIG"
