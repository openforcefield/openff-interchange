"""Utilities for processing and interfacing with the OpenFF Toolkit."""
from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule, Topology


def _get_num_h_bonds(topology: "Topology") -> int:
    """Get the number of (covalent) bonds containing a hydrogen atom."""
    n_bonds_containing_hydrogen = 0

    for bond in topology.bonds:
        if 1 in (bond.atom1.atomic_number, bond.atom2.atomic_number):
            n_bonds_containing_hydrogen += 1

    return n_bonds_containing_hydrogen


def _get_number_excluded_atoms(topology: "Topology", n=3) -> Dict[int, int]:
    exclusions = {a.topology_atom_index: 0 for a in topology.topology_atoms}

    import networkx as nx

    for topology_molecule in topology.topology_molecules:

        mol_graph = topology_molecule.reference_molecule.to_networkx()

        for node_i in mol_graph.nodes:
            for node_j in mol_graph.nodes:
                if node_i >= node_j:
                    continue

                path_length = nx.shortest_path_length(mol_graph, node_i, node_j)

                if path_length <= n:
                    exclusions[
                        node_i + topology_molecule.atom_start_topology_index
                    ] += 1
                    # exclusions[node_j+topology_molecule.atom_start_topology_index] += 1

    return exclusions


def _get_14_pairs(topology_or_molecule: Union["Topology", "Molecule"]):
    """Generate tuples of atom pairs, including symmetric duplicates."""
    # TODO: A replacement of Topology.nth_degree_neighbors in the toolkit
    #       may implement this in the future.
    for bond in topology_or_molecule.bonds:
        atom_i = bond.atom1
        atom_j = bond.atom2
        for atom_i_partner in atom_i.bonded_atoms:
            for atom_j_partner in atom_j.bonded_atoms:
                if atom_i_partner == atom_j_partner:
                    continue

                if atom_i_partner in atom_j_partner.bonded_atoms:
                    continue

                if atom_j_partner in atom_i_partner.bonded_atoms:
                    continue

                if {*atom_i_partner.bonded_atoms}.intersection(
                    {*atom_j_partner.bonded_atoms}
                ):
                    continue

                else:
                    atom_i_partner_index = topology_or_molecule.atom_index(
                        atom_i_partner
                    )
                    atom_j_partner_index = topology_or_molecule.atom_index(
                        atom_j_partner
                    )
                    if atom_i_partner_index > atom_j_partner_index:
                        yield (atom_j_partner, atom_i_partner)
                    else:
                        yield (atom_i_partner, atom_j_partner)
