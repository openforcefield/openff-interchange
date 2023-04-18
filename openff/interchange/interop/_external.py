"""Interfaces to external libraries (without explicitly writing to files)."""
from pathlib import Path
from typing import Union

from openff.interchange import Interchange
from openff.interchange.exceptions import (
    MissingBoxError,
    MissingPositionsError,
    UnsupportedExportError,
)


class InteroperabilityWrapper:
    """Base class for writer wrappers."""

    _write_formats: list[str] = []

    @property
    def write_formats(self) -> list[str]:
        """Return a list of supported writing formats."""
        return self._write_formats


class ParmEdWrapper(InteroperabilityWrapper):
    """Wrapper around ParmEd writers."""

    def __init__(self) -> None:
        self._write_formats = [".gro", ".top", ".prmtop", ".crd", ".inpcrd"]

    def to_file(self, interchange: Interchange, file_path: Union[str, Path]) -> None:
        """
        Convert an Interchange object to a ParmEd Structure and write it to a file.

        """
        if isinstance(file_path, str):
            path = Path(file_path)
        if isinstance(file_path, Path):
            path = file_path

        file_ext = path.suffix.lower()
        if file_ext not in self._write_formats:
            raise UnsupportedExportError(
                f"Writing file format {file_ext} not supported.",
            )

        if interchange.positions is None and file_ext in [".gro", ".crd", ".inpcrd"]:
            raise MissingPositionsError

        if interchange.box is None and file_ext in [".gro"]:
            raise MissingBoxError

        struct = interchange._to_parmed()

        struct.save(path.as_posix(), overwrite=True)
