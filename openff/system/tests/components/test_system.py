from copy import deepcopy

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.stubs import ForceField
from openff.system.tests import BaseTest
from openff.system.tests.energy_tests.openmm import get_openmm_energies


def test_getitem():
    """Test behavior of System.__getitem__"""
    mol = Molecule.from_smiles("CCO")
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(mol.to_topology())

    out.box = [4, 4, 4]

    assert not out.positions
    np.testing.assert_equal(out["box"].m, (4 * np.eye(3) * unit.nanometer).m)
    np.testing.assert_equal(out["box"].m, out["box_vectors"].m)

    assert out["Bonds"] == out.handlers["Bonds"]

    with pytest.raises(LookupError, match="Only str"):
        out[1]

    with pytest.raises(LookupError, match="Could not find"):
        out["CMAPs"]


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
