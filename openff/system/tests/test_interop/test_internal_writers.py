import tempfile

import numpy as np
import parmed as pmd
import pytest
from intermol.gromacs import energies as gmx_energy
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.utils import temporary_cd
from pkg_resources import resource_filename
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.stubs import ForceField
from openff.system.tests.utils import compare_energies


# TODO: Add OC=O
@pytest.mark.parametrize(
    "mol",
    [
        "C",
        "CC",
        "C1COC(=O)O1",
    ],
)
def test_internal_gromacs_writers(mol):
    mol = Molecule.from_smiles(mol)
    mol.name = "FOO"
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    parsley = ForceField("openff_unconstrained-1.0.0.offxml")
    out = parsley.create_openff_system(top)

    out.box = [4, 4, 4] * np.eye(3)
    out.positions = mol.conformers[0] / omm_unit.nanometer

    openmm_sys = parsley.create_openmm_system(top)
    struct = pmd.openmm.load_topology(
        topology=top.to_openmm(),
        system=openmm_sys,
        xyz=out.positions.to(unit.angstrom),
    )
    struct.box = [40, 40, 40, 90, 90, 90]

    with tempfile.TemporaryDirectory() as off_tempdir:
        with temporary_cd(off_tempdir):

            struct.save("reference.top")
            struct.save("reference.gro")

            out.to_top("internal.top", writer="internal")
            out.to_gro("internal.gro", writer="internal")

            compare_gro_files("internal.gro", "reference.gro")
            # TODO: Also compare to out.to_gro("parmed.gro", writer="parmed")

            reference_energy, _ = gmx_energy(
                top="reference.top",
                gro="reference.gro",
                mdp=resource_filename("intermol", "tests/gromacs/grompp.mdp"),
            )

            internal_energy, _ = gmx_energy(
                top="internal.top",
                gro="internal.gro",
                mdp=resource_filename("intermol", "tests/gromacs/grompp.mdp"),
            )

            compare_energies(reference_energy, internal_energy, atol=1e-2)


def compare_gro_files(file1: str, file2: str):
    """Helper function to compare the contents of two GRO files"""
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            # Ignore first two lines and last line
            for line1, line2 in zip(f1.readlines()[2:-1], f2.readlines()[2:-1]):
                # Ignore atom type column
                assert line1[:10] + line1[15:] == line2[:10] + line2[15:]


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

    # TODO: Replace with intermol.gromacs.gmx_energy call after resolving
    #  atomtype name differences that currently force -maxwarn 7
    import os

    from pkg_resources import resource_filename

    mdp_file = resource_filename("intermol", "tests/gromacs/grompp.mdp")
    exit_code = os.system(f"gmx grompp -f {mdp_file} -c out.gro -p out.top -maxwarn 7")

    assert exit_code == 0
