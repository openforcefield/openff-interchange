"""Interfaces with Amber."""
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from openff.toolkit import Topology

    from openff.interchange import Interchange


def from_prmtop(
    file: str,
) -> "Interchange":
    """Import from a prmtop file."""
    from openff.interchange import Interchange

    interchange = Interchange()

    data: Dict[str, List[str]] = dict()

    with open(file) as f:
        for chunk in f.read().split(r"%FLAG"):
            tag, format, *_data = chunk.strip().split()

            if tag == "%VERSION":
                continue

            data[tag] = _data

    interchange.topology = _make_topology(data)

    return interchange


def _make_topology(data: Dict[str, List[str]]) -> "Topology":
    """Make a topology from the data."""
    from openff.toolkit import Topology
    from openff.toolkit.topology._mm_molecule import _SimpleMolecule

    Topology._add_bond = _add_bond

    topology = Topology()

    start_index = 0

    for molecule_index in range(int(data["POINTERS"][11])):
        molecule = _SimpleMolecule()

        end_index = start_index + int(data["ATOMS_PER_MOLECULE"][molecule_index])

        for atom_index in range(start_index, end_index):
            # TODO: Check for isotopes (unsupported) or otherwise botches atomic masses
            molecule.add_atom(
                atomic_number=int(data["ATOMIC_NUMBER"][atom_index]),
                name=data["ATOM_NAME"][atom_index],
            )

        topology.add_molecule(molecule)

        start_index = end_index

    bonds: List[str] = data["BONDS_INC_HYDROGEN"] + data["BONDS_WITHOUT_HYDROGEN"]

    # third value in each triplet is an index to the bond type
    for n1, n2 in zip(bonds[::3], bonds[1::3]):
        # See BONDS_INC_HYDROGEN in https://ambermd.org/prmtop.pdf
        # For run-time efficiency, the atom indexes are actually indexes into a coordinate array,
        # so the actual atom index A is calculated from the coordinate array index N by A = N/3 + 1

        a1 = int(int(n1) / 3)
        a2 = int(int(n2) / 3)

        topology._add_bond(int(a1), int(a2))

    return topology


def _add_bond(self, atom1_index: int, atom2_index: int):
    atom1 = self.atom(atom1_index)
    atom2 = self.atom(atom2_index)

    if atom1.molecule is not atom2.molecule:
        raise ValueError(
            "Cannot add a bond between atoms in different molecules.",
        )

    molecule = atom1.molecule

    molecule.add_bond(
        atom1,
        atom2,
    )
