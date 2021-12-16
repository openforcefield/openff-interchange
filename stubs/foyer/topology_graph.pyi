from typing import Any

class TopologyGraph(object):
    @classmethod
    def from_openff_topology(cls: Any, openff_topology: Any) -> Any: ...
    @classmethod
    def from_parmed(cls: Any, structure: Any) -> Any: ...
