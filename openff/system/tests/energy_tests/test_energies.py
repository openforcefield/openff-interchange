import tempfile

import mbuild as mb
import numpy as np
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils.utils import temporary_cd
from pkg_resources import resource_filename
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies


def test_energies():
    mol = Molecule.from_smiles("CCO")
    mol.name = "FOO"
    top = Topology.from_molecules(5 * [mol])

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    box = [4, 4, 4] * np.eye(3)
    off_sys.box = box

    compound = mb.load("CCO", smiles=True)
    packed_box = mb.fill_box(
        compound=compound, n_compounds=[5], box=mb.Box(box.diagonal())
    )
    positions = packed_box.xyz * unit.nanometer
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

    omm_energies = get_openmm_energies(
        omm_sys,
        positions.m * omm_unit.nanometer,
        box,
        round_positions=3,
    )

    # TODO: Tighten differences
    # np.testing doesn't work on Quantity

    bond_diff = omm_energies["HarmonicBondForce"] - gmx_energies["Bond"]
    assert abs(bond_diff / omm_unit.kilojoules_per_mole) < 1e-3

    angle_diff = omm_energies["HarmonicAngleForce"] - gmx_energies["Angle"]
    assert abs(angle_diff / omm_unit.kilojoules_per_mole) < 1e-3

    torsion_diff = omm_energies["PeriodicTorsionForce"] - gmx_energies["Proper Dih."]
    assert abs(torsion_diff / omm_unit.kilojoules_per_mole) < 1e-3

    # TODO
    # gmx_nonbonded = (
    #     gmx_energies["LJ (SR)"]
    #     + gmx_energies["Disper. corr."]
    #     + gmx_energies["Coulomb (SR)"]
    #     + gmx_energies["Coul. recip."]
    # )
    # nonbonded_diff = omm_energies["NonbondedForce"] - gmx_nonbonded
    # assert abs(nonbonded_diff / omm_unit.kilojoules_per_mole) < 1e-3
