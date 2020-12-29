from pathlib import Path

from openff.system.components.system import System
from openff.system.exceptions import (
    MissingBoxError,
    MissingPositionsError,
    UnsupportedExportError,
)


class InteroperabilityWrapper:
    """Base class for writer wrappers"""

    @property
    def write_formats(self):
        """Return a list of supported writing formats"""
        return self._write_formats


class ParmEdWrapper(InteroperabilityWrapper):
    """Wrapper around ParmEd writers"""

    def __init__(self):
        self._write_formats = [".gro"]

    def to_file(self, openff_sys: System, file_path: Path):
        """
        Convert an OpenFF System to a ParmEd Structure and write it to a file

        """
        file_ext = file_path.suffix.lower()
        if file_ext not in self._write_formats:
            raise UnsupportedExportError(file_ext)

        if openff_sys.positions is None:
            raise MissingPositionsError

        if openff_sys.box is None:
            raise MissingBoxError

        struct = openff_sys.to_parmed()

        struct.save(file_path.as_posix(), overwrite=True)
