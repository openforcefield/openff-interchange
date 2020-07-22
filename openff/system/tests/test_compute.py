import numpy as np
import pytest
from openforcefield.topology import Molecule, Topology

from .. import unit
from ..system import System
from ..tests.base_test import BaseTest
from ..typing.smirnoff.compute import (
    compute_bonds,
    compute_electrostatics,
    compute_potential_energy,
    compute_vdw,
    get_exception_mask,
)


class TestCompute(BaseTest):
    # TODO: Compare these energies to a reference like OpenMM
    @pytest.fixture
    def test_system(self, ethanol_top, parsley):
        return System.from_toolkit(topology=ethanol_top, forcefield=parsley)

    def test_compute_vdw(self, test_system):
        vdw = compute_vdw(test_system)

        assert vdw > 0
        assert vdw.units == unit.Unit("kilocalorie/mole")

    def test_compute_bonds(self, test_system):
        bond = compute_bonds(test_system)

        assert bond > 0
        assert bond.units == unit.Unit("kilocalorie/mole")

    def test_compute_electrostatics(self, test_system):
        electrostatics = compute_electrostatics(test_system)

        assert electrostatics.units == unit.Unit("kilocalorie/mole")

    def test_summing(self, test_system):
        vdw = compute_vdw(test_system)
        bonds = compute_bonds(test_system)
        electrostatics = compute_electrostatics(test_system)

        total = compute_potential_energy(
            test_system, handlers=["Bonds", "vdW", "Electrostatics"]
        )

        assert total == vdw + bonds + electrostatics

    @pytest.mark.parametrize("smiles,mean", [("O", 0.0), ("OO", 0.5 * 2 / 16)])
    def test_exception_mask(self, parsley, smiles, mean):
        """Test that generation of the exception produces the expected values for small systems."""
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol])
        test_system = System.from_toolkit(top, parsley)

        mask = get_exception_mask(test_system)

        assert np.mean(mask) == mean
