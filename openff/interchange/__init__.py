"""A project (and object) for storing, manipulating, and converting molecular mechanics data."""
from openff.interchange._version import get_versions  # type: ignore
from openff.interchange.components.interchange import Interchange

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

__all__ = [
    "Interchange",
]
