from openff.toolkit.topology import Molecule, Topology

from openff.system.components.topology import MMMolecule, MMTopology


def test_from_toolkit():
    mol = Molecule.from_smiles("CCO")
    mmmol = MMMolecule.from_toolkit_molecule(mol)

    assert len(mmmol.atoms) == mol.n_atoms
    assert len(mmmol.bonds) == mol.n_bonds

    top = Topology.from_molecules(2 * [mol])
    mmtop = MMTopology.from_toolkit_topology(top)

    assert len(mmtop.atoms) == top.n_topology_atoms
    assert len(mmtop.bonds) == top.n_topology_bonds
