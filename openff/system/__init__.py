import pint

from openff.system._version import get_versions  # type: ignore

unit = pint.UnitRegistry()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
