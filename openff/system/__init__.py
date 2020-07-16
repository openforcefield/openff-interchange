import pint

from ._version import get_versions

unit = pint.UnitRegistry()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
