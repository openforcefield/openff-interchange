from typing import Any, Optional, NoReturn, Iterator

class Topology(object):
    @classmethod
    def from_openmm(cls, value: Any) -> Topology: ...
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
    ) -> NoReturn: ...
    def add_bond(
        self,
        atom1: Any,
        atom2: Any,
        type: Optional[Any] = None,
        order: Optional[int] = None,
    ) -> NoReturn: ...
    def atom(self, index: int) -> Iterator[Any]: ...
