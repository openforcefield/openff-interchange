import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule

from openff.interchange.components.mdtraj import (
    _combine_topologies,
    _get_num_h_bonds,
    _iterate_pairs,
    _iterate_propers,
    _OFFBioTop,
    _store_bond_partners,
)


@pytest.mark.slow()
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


def test_iterate_pairs_benzene():
    """Check that bonds in rings are not double-counted with _iterate_pairs.
    This should be fixed by using Topology.nth_degree_neighbors directly"""
    benzene = Molecule.from_smiles("c1ccccc1")
    mdtop = md.Topology.from_openmm(benzene.to_topology().to_openmm())

    _store_bond_partners(mdtop)

    assert len({*_iterate_pairs(mdtop)}) == 21


def test_get_num_h_bonds():
    mol = Molecule.from_smiles("CCO")
    top = mol.to_topology()
    mdtop = md.Topology.from_openmm(top.to_openmm())
    assert _get_num_h_bonds(mdtop) == 6


def test_combine_topologies():
    molecule = Molecule.from_smiles("CCO")
    molecule.name = "ETH"

    topology = molecule.to_topology()

    top = _OFFBioTop(mdtop=md.Topology.from_openmm(topology.to_openmm()))

    combined = _combine_topologies(top, top)

    for attr in ("atoms", "bonds", "chains", "residues"):
        attr = "n_" + attr
        assert getattr(combined.mdtop, attr) == 2 * getattr(top.mdtop, attr)
