import pytest
from openff.toolkit import Topology


@pytest.mark.parametrize("collate", [True, False])
def test_collate_or_not_virtual_site_ordering(
    tip4p,
    water,
    collate,
):
    pytest.importorskip("openmm")

    openmm_topology = tip4p.create_interchange(
        Topology.from_molecules([water, water]),
    ).to_openmm_topology(
        collate=collate,
    )

    if collate:
        # particles should be O H H VS O H H VS
        #                     0 1 2 3  4 5 6 7
        virtual_site_indices = (3, 7)

    else:
        # particles should be O H H O H H VS VS
        #                     0 1 2 3 4 5 6  7
        virtual_site_indices = (6, 7)

    for particle_index, particle in enumerate(openmm_topology.atoms()):
        assert (particle.element is None) == (particle_index in virtual_site_indices)
