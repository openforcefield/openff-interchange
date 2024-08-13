from typing import Any, Generator

import numpy
import pandas

class Element:
    atomic_number: int

class Atom:
    name: str
    _bond_partners: list[Atom]
    index: int
    element: Element
    serial: int | None

class Bond:
    atom1: Atom
    atom2: Atom

class Chain: ...

class Residue:
    name: str
    chain: Chain
    resSeq: int | None
    segment_id: str | None
    @property
    def atoms(self) -> Generator[Atom, None, None]: ...

class Topology:
    n_atoms: int
    @classmethod
    def from_openmm(cls, value: Any) -> Topology: ...
    def to_openmm(self) -> Any: ...
    def add_chain(self) -> Any: ...
    def add_residue(
        self,
        name: str,
        chain: Any,
        resSeq: int | None = None,
        segment_id: str | None = None,
    ) -> Any: ...
    def add_atom(
        self,
        name: str,
        element: Any,
        residue: Any,
        serial: int | None = None,
    ): ...
    def add_bond(
        self,
        atom1: Any,
        atom2: Any,
        type: Any | None = None,
        order: int | None = None,
    ): ...
    def atom(self, index: int) -> Atom: ...
    @property
    def atoms(self) -> Generator[Atom, None, None]: ...
    @property
    def bonds(self) -> Generator[Bond, None, None]: ...
    @property
    def residues(self) -> Generator[Residue, None, None]: ...
    def subset(self, atom_indices: numpy.ndarray | list[int]) -> Topology: ...
    def join(self, other: Topology, keep_resSeq: bool = True) -> Topology: ...
    def to_dataframe(self) -> tuple[pandas.DataFrame, numpy.ndarray]: ...
    def residue(self, index: int) -> Residue: ...

class Trajectory:
    topology: Topology
    n_atoms: int
    xyz: numpy.ndarray
    unitcell_lengths: numpy.ndarray
    unitcell_angles: numpy.ndarray
    unitcell_vectors: numpy.ndarray

    def save_pdb(self, file: str): ...

def load_pdb(file: str) -> Trajectory: ...
def load(file: str) -> Trajectory: ...
