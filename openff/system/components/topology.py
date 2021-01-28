from typing import Dict, List, Optional

import ele
from openff.toolkit.topology import Atom, Molecule, Topology
from pydantic import validator

from openff.system import unit
from openff.system.types import DefaultModel, FloatQuantity


class MMAtom(DefaultModel):
    """"""

    name: str = None
    atomic_number: int = None
    element: str = None
    mass: FloatQuantity["atomic_mass_constant"] = None
    extras: Dict[str, str] = dict()

    @validator("mass", always=True)
    def validate_mass(cls, v, values):
        # TODO: handle isotopes
        if v is None:
            if values["element"] is not None:
                element = ele.element_from_symbol(values["atomic_number"])
                return element.mass * unit.atomic_mass_constant
            if values["atomic_number"] is not None:
                element = ele.element_from_atomic_number(values["atomic_number"])
                return element.mass * unit.atomic_mass_constant
            raise ValueError("need a mass")

    @classmethod
    def from_toolkit_atom(cls, toolkit_atom: Atom):
        return cls(
            # name=toolkit_atom.name,
            atomic_number=toolkit_atom.atomic_number,
            # extras={"is_aromatic": str(toolkit_atom.is_aromatic)},
        )


class MMBond(DefaultModel):
    """"""

    atom1_index: int
    atom2_index: int
    order: Optional[float] = None


class MMMolecule(DefaultModel):
    """"""

    name: str = ""
    atoms: List[MMAtom] = list()
    bonds: List[MMBond] = list()
    angles: List[List[MMAtom]] = list()
    propers: List[List[MMAtom]] = list()
    impropers: List[List[MMAtom]] = list()
    extras: Dict[str, str] = dict()

    @classmethod
    def from_toolkit_molecule(cls, toolkit_molecule: Molecule):
        mmmol = cls(name=toolkit_molecule.name)
        for atom in toolkit_molecule.atoms:
            mmmol.atoms.append(MMAtom.from_toolkit_atom(atom))
        for bond in toolkit_molecule.bonds:
            mmmol.bonds.append(
                MMBond(
                    atom1_index=bond.atom1_index,
                    atom2_index=bond.atom2_index,
                )
            )

        return mmmol


class MMTopology(DefaultModel):
    """"""

    atoms: List[MMAtom] = list()
    bonds: List[MMBond] = list()
    molecules: List[MMMolecule] = list()

    @classmethod
    def from_toolkit_topology(cls, toolkit_topology: Topology):
        mmtop = cls()
        for topology_atom in toolkit_topology.topology_atoms:
            mmtop.atoms.append(MMAtom.from_toolkit_atom(topology_atom))

        for bond in toolkit_topology.topology_bonds:
            atom1, atom2 = bond.atoms
            atom1_idx, atom2_idx = atom1.topology_atom_index, atom2.topology_atom_index
            mmtop.bonds.append(
                MMBond(
                    atom1_index=atom1_idx,
                    atom2_index=atom2_idx,
                ),
            )

        # TODO: Process molecules, resisdues/chains, other valence terms

        return mmtop
