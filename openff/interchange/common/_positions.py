from typing import Optional

import numpy
from openff.toolkit import Quantity, Topology


def _infer_positions(
    topology: Topology,
    positions: Optional[Quantity] = None,
) -> Optional[Quantity]:
    if positions is not None:
        return positions

    for molecule in topology.molecules:
        if molecule.n_conformers == 0:
            # if _any_ molecule lacks conformers, break out immediately
            return None

    return numpy.concatenate(
        [molecule.conformers[0] for molecule in topology.molecules],
    )
