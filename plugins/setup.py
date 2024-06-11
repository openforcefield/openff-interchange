"""
Plugins for custom SMIRNOFF types.
"""

from setuptools import setup

setup(
    name="nonbonded_plugins",
    packages=["nonbonded_plugins"],
    version="0.0.0",
    include_package_data=True,
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "BuckinghamHandler=nonbonded_plugins.nonbonded:BuckinghamHandler",
            "BuckinghamVirtualSiteHandler=nonbonded_plugins.virtual_sites:BuckinghamVirtualSiteHandler",
            "DoubleExponentialHandler=nonbonded_plugins.nonbonded:DoubleExponentialHandler",
            "C4IonHandler=nonbonded_plugins.nonbonded:C4IonHandler",
        ],
        "openff.interchange.plugins.collections": [
            "BuckinghamCollection=nonbonded_plugins.nonbonded:SMIRNOFFBuckinghamCollection",
            "BuckinghamVirtualSiteCollection=nonbonded_plugins.virtual_sites:SMIRNOFFBuckinghamVirtualSiteCollection",
            "DoubleExponentialCollection=nonbonded_plugins.nonbonded:SMIRNOFFDoubleExponentialCollection",
            "C4IonCollection=nonbonded_plugins.nonbonded:SMIRNOFFC4IonCollection",
        ],
    },
)
