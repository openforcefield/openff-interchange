from typing import Union

import openmm
import openmm.app
from openff.toolkit import Topology

from openff.interchange.exceptions import UnsupportedImportError


def _check_compatible_inputs(
    system: openmm.System,
    topology: Union[openmm.app.Topology, Topology, None],
):
    """Check that inputs are compatible and supported."""
    for index in range(system.getNumParticles()):
        if system.isVirtualSite(index):
            raise UnsupportedImportError(
                "A particle is a virtual site, which is not yet supported.",
            )

    if isinstance(topology, Topology):
        _topology = topology.to_openmm()
    else:
        _topology = topology

    if _topology is not None:
        if system.getNumParticles() != _topology.getNumAtoms():
            raise UnsupportedImportError(
                f"The number of particles in the system ({system.getNumParticles()}) and "
                f"the number of atoms in the topology ({_topology.getNumAtoms()}) do not match.",
            )
