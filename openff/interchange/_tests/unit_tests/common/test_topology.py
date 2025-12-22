"""Unit tests for `InterchangeTopology`."""

import pytest

from openff.interchange.common._topology import InterchangeTopology
from openff.interchange.exceptions import NoPositionsError


@pytest.mark.parametrize("conversion_method", ["json", "dict"])
def test_from_toolkit_topology(alanine_dipeptide, conversion_method):
    match conversion_method:
        case "json":
            topology = InterchangeTopology.from_json(
                alanine_dipeptide.to_json(),
            )
        case "dict":
            topology = InterchangeTopology.from_dict(
                alanine_dipeptide.to_dict(),
            )

    assert topology.n_atoms == alanine_dipeptide.n_atoms
    assert topology.n_bonds == alanine_dipeptide.n_bonds
    assert topology.n_molecules == alanine_dipeptide.n_molecules

    for atom_a, atom_b in zip(topology.atoms, alanine_dipeptide.atoms):
        assert atom_a.atomic_number == atom_b.atomic_number

    for exception in [NoPositionsError, NotImplementedError]:
        with pytest.raises(exception):
            topology.get_positions()

        with pytest.raises(exception):
            topology.clear_positions()

        with pytest.raises(exception):
            topology.set_positions(alanine_dipeptide.molecule(0).conformers[0])
