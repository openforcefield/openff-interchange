import tempfile

import numpy as np
import pytest
from intermol.gromacs import energies as gmx_energy
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.utils import temporary_cd
from pkg_resources import resource_filename
from simtk import unit as omm_unit

from openff.system.stubs import ForceField
from openff.system.tests.utils import compare_energies


# TODO: Add CC, OC=O, CCOC, C1COC(=O)O1, more
@pytest.mark.parametrize(
    "mol",
    [
        "C",
    ],
)
def test_internal_gromacs_writers(mol):
    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(top)

    out.box = [4, 4, 4] * np.eye(3)
    out.positions = mol.conformers[0] / omm_unit.nanometer

    with tempfile.TemporaryDirectory() as off_tempdir:
        with temporary_cd(off_tempdir):
            out.to_gro("internal.gro", writer="internal")
            out.to_gro("parmed.gro", writer="parmed")

            compare_gro_files("internal.gro", "parmed.gro")

            out.to_top("internal.top", writer="internal")
            out.to_top("parmed.top", writer="parmed")

            pmd_energy, _ = gmx_energy(
                top="parmed.top",
                gro="parmed.gro",
                mdp=resource_filename("intermol", "tests/gromacs/grompp.mdp"),
            )

            internal_energy, _ = gmx_energy(
                top="internal.top",
                gro="internal.gro",
                mdp=resource_filename("intermol", "tests/gromacs/grompp.mdp"),
            )

            try:
                compare_energies(pmd_energy, internal_energy)
            except Exception:
                import os

                os.system("cp * /Users/mwt/software/openff-system/tmp/")
                raise Exception


def compare_gro_files(file1: str, file2: str):
    """Helper function to compare the contents of two GRO files"""
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            # Ignore first two lines and last line
            assert f1.readlines()[2:-1] == f2.readlines()[2:-1]


def test_sanity_grompp():
    """Basic test to ensure that a topology can be processed without errors"""
    mol = Molecule.from_smiles("CC")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()

    parsley = ForceField("openff-1.0.0.offxml")
    off_sys = parsley.create_openff_system(top)

    off_sys.box = [4, 4, 4] * np.eye(3)
    off_sys.positions = mol.conformers[0] / omm_unit.angstrom

    off_sys.to_gro("out.gro", writer="internal")
    off_sys.to_top("out.top", writer="internal")

    ener, _ = gmx_energy(
        top="out.top",
        gro="out.gro",
        mdp=resource_filename("intermol", "tests/gromacs/grompp.mdp"),
    )

    # Just check that nothing is read as NaN
    for num in ener.values():
        assert not np.isnan(num / num.unit)
