from openff.toolkit import ForceField, Molecule
from packaging.version import Version

from openff.interchange import __version__
from openff.interchange.drivers import get_openmm_energies


def test_versions_sane():
    # partially safeguard against a malformed version number;
    # 0.minor.patch and major.0.patch should be okay but not 0.0.patch
    found_version = Version(__version__)
    assert not (found_version.minor == 0 and found_version.major == 0), found_version

    print(f"found version {found_version}")


def test_build_interchange_and_get_openmm_energy():
    molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    molecule.generate_conformers()

    interchange = ForceField("openff-2.3.0.offxml").create_interchange(molecule.to_topology())

    get_openmm_energies(interchange)
