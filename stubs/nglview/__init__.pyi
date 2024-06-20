from typing import Iterable, Union

class NGLWidget(object):
    ...
    def add_representation(
        self,
        repr_type: str,
        selection: Union[str, Iterable] = "all",
        **kwargs,
    ): ...

def show_file(file_path: str) -> NGLWidget: ...
