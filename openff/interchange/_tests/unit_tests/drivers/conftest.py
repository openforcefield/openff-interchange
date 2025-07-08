"""Fixtures for driver tests."""

from copy import deepcopy

import pytest
from openff.toolkit import Quantity

from openff.interchange._tests import MoleculeWithConformer


@pytest.fixture
def methane_dimer(sage):
    molecule = MoleculeWithConformer.from_smiles("C")
    topology = deepcopy(molecule).to_topology()

    molecule._conformers[0] += Quantity([4, 0, 0], "angstrom")

    topology.add_molecule(molecule)
    topology.box_vectors = Quantity([3, 3, 3], "nanometer")

    interchange = sage.create_interchange(topology)
    interchange.minimize()

    return interchange
