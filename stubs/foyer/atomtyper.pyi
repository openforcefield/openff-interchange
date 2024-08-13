from typing import Any, TypedDict

class _AtomTypeInfo(TypedDict):
    whitelist: set[str]
    blacklist: set[str]
    atomtype: str

def find_atomtypes(structure: Any, forcefield: Any, max_iter: int = 10) -> dict[int, _AtomTypeInfo]: ...
