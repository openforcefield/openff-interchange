"""A project (and object) for storing, manipulating, and converting molecular mechanics data."""

import importlib
from importlib.metadata import version
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checkers can't see lazy-imported objects
    from openff.interchange.components.interchange import Interchange


__version__ = version("openff.interchange")

__all__: list[str] = [
    "Interchange",
]

_objects: dict[str, str] = {
    "Interchange": "openff.interchange.components.interchange",
}


def __getattr__(name) -> ModuleType:
    """
    Lazily import objects from submodules.

    Taken from openff/toolkit/__init__.py
    """
    module = _objects.get(name)
    if module is not None:
        try:
            return importlib.import_module(module).__dict__[name]
        except ImportError as error:
            raise ImportError from error

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Add _objects to dir()."""
    keys = (*globals().keys(), *_objects.keys())
    return sorted(keys)
