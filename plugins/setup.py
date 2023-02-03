"""
Plugins for custom SMIRNOFF types.
"""
from setuptools import setup

setup(
    name="nonbonded_plugins",
    py_modules="nonbonded_plugins",
    include_package_data=True,
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "BuckinghamHandler = nonbonded_plugins:BuckinghamHandler",
        ],
        "openff.interchange.plugins.handlers": [
            "BuckinghamCollection = nonbonded_plugins:BuckinghamCollection",
        ],
    },
)
