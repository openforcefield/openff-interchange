from typing import Any, Sequence

class Particle:
    def __init__(self, name: str | None) -> None: ...

class Compound:
    def __init__(self, name: str = "Compound", subcompounds: list | None = None) -> None: ...
    name: str
    xyz: Any
    def add(
        self,
        new_child: Any,
        label: str | None = None,
        containment: bool = True,
        replace: bool = False,
        inherit_periodicity: bool = True,
        inherit_box: bool = False,
        reset_rigid_ids: bool = True,
    ) -> None: ...
    def add_bond(self, particle_pair: Sequence) -> None: ...
    def __getitem__(self, selection: int) -> Compound: ...
