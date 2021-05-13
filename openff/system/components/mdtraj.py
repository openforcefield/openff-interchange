import mdtraj as md
from openff.toolkit.topology import Topology


class OFFBioTop(Topology):
    def __init__(self, mdtop=None, *args, **kwargs):
        self.mdtop = mdtop
        super().__init__(*args, **kwargs)


def _store_bond_partners(mdtop):
    for atom in mdtop.atoms:
        atom._bond_partners = []
    for bond in mdtop.bonds:
        bond.atom1._bond_partners.append(bond.atom2)
        bond.atom2._bond_partners.append(bond.atom1)


def _iterate_angles(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                if atom1.index < atom3.index:
                    yield (atom1, atom2, atom3)
                else:
                    # Do no duplicate
                    pass  # yield (atom3, atom2, atom1)


def _iterate_propers(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                for atom4 in atom3._bond_partners:
                    if atom4 in (atom1, atom2):
                        continue

                    if atom1.index < atom4.index:
                        yield (atom1, atom2, atom3, atom4)
                    else:
                        # Do no duplicate
                        pass  # yield (atom4, atom3, atom2, atom1)


def _iterate_impropers(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                for atom4 in atom2._bond_partners:
                    if atom4 in (atom3, atom1):
                        continue

                    yield (atom1, atom2, atom3, atom4)


def _iterate_pairs(mdtop):
    for bond in mdtop.bonds:
        atom_i = bond.atom1
        atom_j = bond.atom2
        for atom_i_partner in atom_i._bond_partners:
            for atom_j_partner in atom_j._bond_partners:
                if atom_i_partner == atom_j_partner:
                    continue

                if atom_i_partner in atom_j_partner._bond_partners:
                    continue

                if atom_j_partner in atom_i_partner._bond_partners:
                    continue

                if {*atom_i_partner._bond_partners}.intersection(
                    {*atom_j_partner._bond_partners}
                ):
                    continue

                else:
                    yield (atom_i_partner, atom_j_partner)


def _get_num_h_bonds(mdtop):
    """Get the number of (covalent) bonds containing a hydrogen atom"""
    n_bonds_containing_hydrogen = 0

    for bond in mdtop.bonds:
        if md.element.hydrogen in (bond.atom1.element, bond.atom2.element):
            n_bonds_containing_hydrogen += 1

    return n_bonds_containing_hydrogen
