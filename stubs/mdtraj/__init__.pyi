from typing import Any, Generator, Optional, Union

import numpy
import pandas

class Element(object):
    atomic_number: int

class Atom(object):
    name: str
    _bond_partners: list[Atom]
    index: int
    element: Element
    serial: Optional[int]

class Bond(object):
    atom1: Atom
    atom2: Atom

class Chain(object): ...

class Residue(object):
    name: str
    chain: Chain
    resSeq: Optional[int]
    segment_id: Optional[str]
    @property
    def atoms(self) -> Generator[Atom, None, None]: ...

class Topology(object):
    n_atoms: int
    @classmethod
    def from_openmm(cls, value: Any) -> Topology: ...
    def to_openmm(self) -> Any: ...
    def add_chain(self) -> Any: ...
    def add_residue(
        self,
        name: str,
        chain: Any,
        resSeq: Optional[int] = None,
        segment_id: Optional[str] = None,
    ) -> Any: ...
    def add_atom(
        self,
        name: str,
        element: Any,
        residue: Any,
        serial: Optional[int] = None,
    ): ...
    def add_bond(
        self,
        atom1: Any,
        atom2: Any,
        type: Optional[Any] = None,
        order: Optional[int] = None,
    ): ...
    def atom(self, index: int) -> Atom: ...
    @property
    def atoms(self) -> Generator[Atom, None, None]: ...
    @property
    def bonds(self) -> Generator[Bond, None, None]: ...
    @property
    def residues(self) -> Generator[Residue, None, None]: ...
    def subset(self, atom_indices: Union[numpy.ndarray, list[int]]) -> Topology: ...
    def join(self, other: Topology, keep_resSeq: bool = True) -> Topology: ...
    def to_dataframe(self) -> tuple[pandas.DataFrame, numpy.ndarray]: ...
    def residue(self, index: int) -> Residue: ...

class Trajectory(object):
    topology: Topology
    n_atoms: int
    unitcell_lengths: numpy.ndarray
    unitcell_angles: numpy.ndarray

    def save_pdb(self, file: str): ...

def load_pdb(file: str) -> Trajectory: ...
def load(file: str) -> Trajectory: ...
