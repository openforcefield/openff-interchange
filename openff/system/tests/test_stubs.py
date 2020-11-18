from openforcefield.topology import Molecule, Topology

from openff.system.tests.base_test import BaseTest


class TestStubs(BaseTest):
    """Test the functionality of the stubs.py module"""

    def test_from_parsley(self, parsley):
        top = Topology.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = parsley.create_openff_system(top)

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()
