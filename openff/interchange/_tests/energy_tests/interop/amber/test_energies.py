import pytest
from openff.toolkit import Quantity
from openff.utilities import skip_if_missing

from openff.interchange._tests import MoleculeWithConformer, requires_ambertools
from openff.interchange.drivers import get_amber_energies, get_openmm_energies


@requires_ambertools
@skip_if_missing("openmm")
@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1", "C1#CC#CC#C1", "C1=CC=C2C(=C1)C=CC3=C2C=CC4=CC=CC=C43"],
)
def test_polycyclic_nonbonded(smiles, sage_unconstrained):
    molecule = MoleculeWithConformer.from_smiles(smiles)
    topology = molecule.to_topology()
    topology.box_vectors = Quantity([4, 4, 4], "nanometer")

    interchange = sage_unconstrained.create_interchange(topology)
    openmm_vdw = get_openmm_energies(
        interchange,
        combine_nonbonded_forces=False,
    ).energies["vdW"]
    amber_vdw = get_amber_energies(interchange).energies["vdW"]

    assert abs(openmm_vdw - amber_vdw).m < 1e-3
