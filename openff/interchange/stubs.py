"""
Monkeypatching external classes with custom functionality
"""
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import OFFBioTop


def _create_openff_interchange(
    self,
    topology: OFFBioTop,
    box=None,
) -> Interchange:

    return Interchange.from_smirnoff(self, topology, box)


ForceField.create_openff_interchange = _create_openff_interchange
