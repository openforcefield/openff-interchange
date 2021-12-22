from typing import Any, Optional

class TopologyGraph(object):
    @classmethod
    def from_openff_topology(cls: Any, openff_topology: Any) -> Any: ...
    def add_atom(
        self,
        index: int,
        name: str,
        atomic_number: Optional[int] = None,
        element: Optional[Any] = None,
        **kwargs
    ) -> None: ...
    def add_bond(self, atom_1_index: int, atom_2_index: int) -> None: ...
