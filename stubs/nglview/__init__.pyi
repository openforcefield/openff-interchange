from typing import Any, Iterable, Union

class NGLWidget(object):
    def __init__(
        self, structure: Any, representations: Any, parameters: Any = None, **kwargs
    ): ...
    def add_representation(
        self,
        repr_type: str,
        selection: Union[str, Iterable] = "all",
        **kwargs,
    ): ...

def show_file(file_path: str) -> NGLWidget: ...
