from typing import Any

class TopologyGraph:
    @classmethod
    def from_openff_topology(cls: Any, openff_topology: Any) -> Any: ...
    def add_atom(
        self,
        index: int,
        name: str,
        atomic_number: int | None = None,
        element: Any | None = None,
        **kwargs,
    ) -> None: ...
    def add_bond(self, atom_1_index: int, atom_2_index: int) -> None: ...
    @classmethod
    def from_parmed(cls: Any, structure: Any) -> Any: ...
