"""
Monkeypatching external classes with custom functionality
"""
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System


def _create_openff_system(
    self,
    topology: OFFBioTop,
    box=None,
) -> System:

    return System.from_smirnoff(self, topology, box)


ForceField.create_openff_system = _create_openff_system
