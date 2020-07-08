"""
openff-system
A molecular system object from the Open Force Field Initiative
"""

import pint

unit = pint.UnitRegistry()

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
