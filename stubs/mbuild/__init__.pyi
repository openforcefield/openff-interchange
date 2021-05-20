from typing import Optional, List, Any

class Particle(object):
    def __init__(self, name: Optional[str]) -> None: ...

class Compound(object):
    def __init__(
        self, name: str="Compound", subcompounds: Optional[List] = None
    ) -> None: ...

    name: str
    xyz: Any
    def add(
        self,
        new_child: Any,
        label: Optional[str] = None,
        containment: bool = True,
        replace: bool = False,
        inherit_periodicity: bool = True,
        inherit_box: bool = False,
        reset_rigid_ids: bool = True,
    ) -> None: ...

    def add_bond(self, Sequence) -> None: ...

    def __getitem__(self, int) -> Compound: ...
