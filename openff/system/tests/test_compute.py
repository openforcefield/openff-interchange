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
)


class TestCompute(BaseTest):
    # TODO: Compare these energies to a reference like OpenMM
    @pytest.fixture
    def hexanone(self, parsley):
        mol = Molecule.from_smiles("CCCC(=O)C")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules(mol)
        return System.from_toolkit(topology=top, forcefield=parsley)

    def test_compute_vdw(self, hexanone):
        vdw = compute_vdw(hexanone)

        assert vdw > 0
        assert vdw.units == unit.Unit("kilocalorie/mole")

    def test_compute_bonds(self, hexanone):
        bond = compute_bonds(hexanone)

        assert bond > 0
        assert bond.units == unit.Unit("kilocalorie/mole")

    def test_compute_electrostatics(self, hexanone):
        electrostatics = compute_electrostatics(hexanone)

        assert electrostatics.units == unit.Unit("kilocalorie/mole")

    def test_summing(self, hexanone):
        vdw = compute_vdw(hexanone)
        bonds = compute_bonds(hexanone)
        electrostatics = compute_electrostatics(hexanone)

        total = compute_potential_energy(
            hexanone, handlers=["Bonds", "vdW", "Electrostatics"]
        )

        assert total == vdw + bonds + electrostatics
