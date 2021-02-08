import mbuild as mb
import numpy as np
from openff.toolkit.topology import Molecule, Topology

from openff.system import unit
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies
from openff.system.tests.energy_tests.utils import compare_gromacs_openmm


def test_energies():
    mol = Molecule.from_smiles("CCO")
    mol.name = "FOO"
    top = Topology.from_molecules(50 * [mol])

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    box = [4, 4, 4] * np.eye(3)
    off_sys.box = box

    compound = mb.load("CCO", smiles=True)
    packed_box = mb.fill_box(
        compound=compound, n_compounds=[50], box=mb.Box(box.diagonal())
    )
    positions = packed_box.xyz * unit.nanometer
    off_sys.positions = positions

    gmx_energies = get_gromacs_energies(off_sys)
    omm_energies = get_openmm_energies(off_sys, round_positions=3)

    compare_gromacs_openmm(omm_energies=omm_energies, gmx_energies=gmx_energies)
