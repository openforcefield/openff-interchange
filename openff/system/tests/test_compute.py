from openforcefield.topology import Topology, Molecule

from ..typing.smirnoff.compute import compute_vdw
from ..system import System
from ..tests.base_test import BaseTest


class TestCompute(BaseTest):

    def test_compute_vdw(self, parsley):
        mol = Molecule.from_smiles('CCCCC(=O)C')
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules(mol)
        hexanone = System.from_toolkit(topology=top, forcefield=parsley)

        compute_vdw(hexanone)
