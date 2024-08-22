from typing import Iterable

class NGLWidget:
    def __init__(self,
                 structure=None,
                 representations=None,
                 parameters=None,
                 **kwargs):...
    def add_representation(
        self,
        repr_type: str,
        selection: str | Iterable,
        **kwargs,
    ): ...

def show_file(file_path: str) -> NGLWidget: ...
