import pytest
from openff.toolkit import Topology


@pytest.mark.parametrize("name_sol", [True, False])
@pytest.mark.parametrize("collate", [True, False])
def test_collate_or_not_virtual_site_ordering(
    tip4p,
    water,
    name_sol,
    collate,
):
    pytest.importorskip("openmm")

    if name_sol:
        for atom in water.atoms:
            atom.metadata["residue_name"] = "SOL"

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

    assert {residue.name for residue in openmm_topology.residues()} == {"SOL"} if name_sol else {"UNK"}


def test_each_molecule_with_virtual_sites_has_its_own_funnky_virtual_site_residue(
    tip5p,
    water,
):
    topology = Topology.from_molecules([water, water])

    for index, molecule in enumerate(topology.molecules):
        for atom in molecule.atoms:
            atom.metadata["residue_name"] = "SOL"
            atom.metadata["residue_number"] = str(index)

    openmm_topology = tip5p.create_interchange(topology).to_openmm_topology()

    assert openmm_topology.getNumResidues() == 4

    assert [3, 3, 2, 2] == [len([*residue.atoms()]) for residue in openmm_topology.residues()]
