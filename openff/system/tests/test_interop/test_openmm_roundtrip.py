import pytest
from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.interop.openmm import from_openmm
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.openmm import get_openmm_energies


@pytest.mark.slow
def test_openmm_roundtrip():
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    off_sys.box = [4, 4, 4]
    off_sys.positions = mol.conformers[0].value_in_unit(omm_unit.nanometer)

    omm_sys = off_sys.to_openmm()

    converted = from_openmm(
        system=omm_sys,
    )

    converted.topology = off_sys.topology
    converted.box = off_sys.box
    converted.positions = off_sys.positions

    get_openmm_energies(off_sys).compare(
        get_openmm_energies(converted),
        custom_tolerances={"Nonbonded": 1.5 * omm_unit.kilojoule_per_mole},
    )
