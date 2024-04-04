import pytest
from openff.toolkit import Quantity

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.components._packmol import (
    _find_packmol,
    solvate_topology_nonwater,
)
from openff.interchange.drivers import get_openmm_energies


@pytest.mark.slow()
@pytest.mark.skipif(_find_packmol() is None, reason="PACKMOL not found")
def test_solvate_ligand_in_nonwater(sage):
    """
    Test that a ligand can be solvated in a non-water solvent with a sane
    starting point, like setting up a SFE run.

    See uses of `solvate_topology` somewhere in openfe_skunkworks
    """
    pytest.importorskip("openmm")

    # ibuprofen in hexanol
    ligand = MoleculeWithConformer.from_smiles(
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        allow_undefined_stereo=True,
    )
    solvent = MoleculeWithConformer.from_smiles("CCCCCCO")

    topology = solvate_topology_nonwater(
        topology=ligand.to_topology(),
        solvent=solvent,
        padding=Quantity(0.8, "nanometer"),
        target_density=Quantity(800, "kilogram / meter ** 3"),
    )

    interchange = sage.create_interchange(topology)

    packed_energy = get_openmm_energies(interchange)

    interchange.minimize(force_tolerance=Quantity(100, "kJ / mol / nm"))

    assert get_openmm_energies(interchange).total_energy < packed_energy.total_energy
