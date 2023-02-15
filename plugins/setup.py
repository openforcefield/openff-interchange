"""
Plugins for custom SMIRNOFF types.
"""
from setuptools import setup

setup(
    name="nonbonded_plugins",
    version="0.0.0",
    include_package_data=True,
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "BuckinghamHandler = nonbonded_plugins:BuckinghamHandler",
        ],
        "openff.interchange.plugins.collections": [
            "BuckinghamCollection = nonbonded_plugins:SMIRNOFFBuckinghamCollection",
        ],
    },
)
