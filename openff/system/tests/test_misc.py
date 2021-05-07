import mdtraj as md
from openff.toolkit.topology import Molecule

from openff.system.components.misc import (
    _get_num_h_bonds,
    _iterate_pairs,
    _iterate_propers,
    _store_bond_partners,
)


def test_iterate_pairs():
    mol = Molecule.from_smiles("C1#CC#CC#C1")

    top = mol.to_topology()

    mdtop = md.Topology.from_openmm(top.to_openmm())

    _store_bond_partners(mdtop)
    pairs = {
        tuple(sorted((atom1.index, atom2.index)))
        for atom1, atom2 in _iterate_pairs(mdtop)
    }
    assert len(pairs) == 3
    assert len([*_iterate_propers(mdtop)]) > len(pairs)


def test_get_num_h_bonds():
    mol = Molecule.from_smiles("CCO")
    top = mol.to_topology()
    mdtop = md.Topology.from_openmm(top.to_openmm())
    assert _get_num_h_bonds(mdtop) == 6
