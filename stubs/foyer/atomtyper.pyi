
from typing import Any, Dict, Set, Union, TypedDict

class _AtomTypeInfo(TypedDict):
    whitelist: Set[str]
    blacklist: Set[str]
    atomtype: str


def find_atomtypes(
    structure: Any, forcefield: Any, max_iter: int = 10
) -> Dict[int, _AtomTypeInfo]: ...
