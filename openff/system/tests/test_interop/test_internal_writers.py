import tempfile

import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils.utils import temporary_cd
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import (
    get_gromacs_energies,
    get_mdp_file,
    run_gmx_energy,
)


# TODO: Add OC=O
@pytest.mark.parametrize(
    "mol",
    [
        "C",
        "CC",
        # "OC=O",  # ParmEd conversion untrustworthy, see #91
        "CCOC",
        "C1COC(=O)O1",  # TODO: These dihedrals are off by ~3 kJ/mol
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

            reference_energy = run_gmx_energy(
                top_file="reference.top",
                gro_file="reference.gro",
                mdp_file=get_mdp_file("default"),
            )

            internal_energy = run_gmx_energy(
                top_file="internal.top",
                gro_file="internal.gro",
                mdp_file=get_mdp_file("default"),
            )

            reference_energy.compare(
                internal_energy,
                custom_tolerances={"Bond": 2e-2 * omm_unit.kilojoule_per_mole},
            )


def compare_gro_files(file1: str, file2: str):
    """Helper function to compare the contents of two GRO files"""
    with open(file1) as f1:
        with open(file2) as f2:
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
    exit_code = os.system(f"gmx grompp -f {mdp_file} -c out.gro -p out.top -maxwarn 1")

    assert exit_code == 0


def test_water_dimer():
    """Test that a water dimer can be written and the files can be grommp'd"""
    from openff.system.utils import get_test_file_path

    tip3p = ForceField(get_test_file_path("tip3p.offxml"))
    water = Molecule.from_smiles("O")
    top = Topology.from_molecules(2 * [water])

    from simtk import openmm
    from simtk import unit as omm_unit

    pdbfile = openmm.app.PDBFile(get_test_file_path("water-dimer.pdb"))

    positions = np.array(pdbfile.positions / omm_unit.nanometer) * unit.nanometer

    openff_sys = tip3p.create_openff_system(top)
    openff_sys.positions = positions
    openff_sys.box = [10, 10, 10] * unit.nanometer

    gmx_energies = get_gromacs_energies(openff_sys, writer="internal")

    assert gmx_energies is not None
