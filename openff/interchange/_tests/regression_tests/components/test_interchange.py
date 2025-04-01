from openff.toolkit import Topology

from openff.interchange._tests import MoleculeWithConformer


def compare_topologies(topology1, topology2):
    """Compare two topologies for equality."""
    assert topology1.to_dict() == topology2.to_dict()


class TestPDBRoundtripVsToolkit:
    def test_roundtrip_vs_toolkit(self, sage, tmp_path):
        """Test that Interchange.to_pdb can roundtrip with Topology.from_pdb."""
        h2po4 = MoleculeWithConformer.from_smiles("[O-]P(=O)(O)O")

        out = sage.create_interchange(h2po4.to_topology())

        # include_virtual_sites=True errors since there are no virtual sites
        out.to_pdb("interchange.pdb", include_virtual_sites=False)
        h2po4.to_topology().to_file("toolkit.pdb")

        compare_topologies(
            Topology.from_pdb("interchange.pdb", unique_molecules=[h2po4]),
            Topology.from_pdb("toolkit.pdb", unique_molecules=[h2po4]),
        )
