import tempfile

import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.utils import temporary_cd
from pkg_resources import resource_filename
from simtk import unit

from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies


def test_energies():
    mol = Molecule.from_smiles("CCO")
    mol.name = "FOO"
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    positions = mol.conformers[0].in_units_of(unit.nanometer) / unit.nanometer
    box = [4, 4, 4] * np.eye(3)

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)
    off_sys.box = box
    off_sys.positions = positions

    omm_sys = off_sys.to_openmm()

    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):

            mdp_file = resource_filename("intermol", "tests/gromacs/grompp.mdp")
            off_sys.to_gro("out.gro", writer="internal")
            off_sys.to_top("out.top", writer="internal")
            gmx_energies, _ = get_gromacs_energies(
                top="out.top",
                gro="out.gro",
                mdp=mdp_file,
            )

    omm_energies = get_openmm_energies(omm_sys, positions, box)

    # TODO: Tighten differences
    # np.testing doesn't work on Quantity
    assert (
        omm_energies["HarmonicBondForce"] - gmx_energies["Bond"]
    ) / unit.kilojoule_per_mole < 1e-2
    assert (
        omm_energies["HarmonicAngleForce"] - gmx_energies["Angle"]
    ) / unit.kilojoule_per_mole < 2e-1
