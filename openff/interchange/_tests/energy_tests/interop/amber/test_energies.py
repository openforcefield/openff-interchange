import pytest
from openff.toolkit import Quantity
from openff.utilities import skip_if_missing

from openff.interchange._tests import MoleculeWithConformer, requires_ambertools
from openff.interchange.constants import kcal_mol
from openff.interchange.drivers import get_amber_energies, get_openmm_energies


@requires_ambertools
@skip_if_missing("openmm")
@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1", "C1#CC#CC#C1", "C1=CC=C2C(=C1)C=CC3=C2C=CC4=CC=CC=C43"],
)
def test_polycyclic_nonbonded(smiles, sage_no_switch):
    molecule = MoleculeWithConformer.from_smiles(smiles)
    topology = molecule.to_topology()
    topology.box_vectors = Quantity([10, 10, 10], "nanometer")

    interchange = sage_no_switch.create_interchange(topology)

    openmm_vdw = get_openmm_energies(
        interchange,
        combine_nonbonded_forces=False,
    ).energies["vdW"]

    amber_vdw = get_amber_energies(interchange).energies["vdW"]

    assert openmm_vdw.m == pytest.approx(
        amber_vdw.m,
        rel=1e-5,
    ), f"{openmm_vdw.m_as(kcal_mol)=}, {amber_vdw.m_as(kcal_mol)=}"
