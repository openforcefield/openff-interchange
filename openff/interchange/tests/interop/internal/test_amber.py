import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers import get_amber_energies, get_openmm_energies
from openff.interchange.tests.utils import needs_gmx

kj_mol = unit.kilojoule / unit.mol


@skip_if_missing("intermol")
@needs_gmx
@pytest.mark.slow()
def test_amber_energy():
    """Basic test to see if the amber energy driver is functional"""
    mol = Molecule.from_smiles(r"Cl/CC\Cl")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    top.mdtop = md.Topology.from_openmm(top.to_openmm())

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")
    off_sys = Interchange.from_smirnoff(parsley, top)

    off_sys.box = [4, 4, 4]
    off_sys.positions = mol.conformers[0]

    omm_energies = get_openmm_energies(off_sys)
    amb_energies = get_amber_energies(off_sys)

    omm_energies.compare(
        amb_energies,
        custom_tolerances={
            "Bond": 0.02 * kj_mol,
            "Angle": 0.01 * kj_mol,
            "Nonbonded": 0.01 * kj_mol,
        },
    )
