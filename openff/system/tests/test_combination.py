from copy import deepcopy

import mdtraj as md
import numpy as np
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.openmm import get_openmm_energies


class TestSystemCombination(BaseTest):
    def test_basic_combination(self):
        """Test basic use of System.__add__() based on the README example"""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = OFFBioTop.from_molecules([mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        parsley = ForceField("openff_unconstrained-1.0.0.offxml")
        openff_sys = parsley.create_openff_system(top)

        openff_sys.box = [4, 4, 4] * np.eye(3)
        openff_sys.positions = mol.conformers[0]._value / 10.0

        # Copy and translate atoms by [1, 1, 1]
        other = System()
        other._inner_data = deepcopy(openff_sys._inner_data)
        other.positions += 1.0 * unit.nanometer

        combined = openff_sys + other

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)
