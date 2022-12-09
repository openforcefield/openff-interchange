import pytest
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange.components.toolkit import (
    _check_electrostatics_handlers,
    _combine_topologies,
    _get_14_pairs,
    _get_num_h_bonds,
)
from openff.interchange.tests import _BaseTest


class TestToolkitUtils(_BaseTest):
    @pytest.mark.parametrize(
        ("smiles", "num_pairs"),
        [("C#C", 1), ("CCO", 12), ("C1=CC=CC=C1", 24)],
    )
    def test_get_14_pairs(self, smiles, num_pairs):
        mol = Molecule.from_smiles(smiles)
        assert len([*_get_14_pairs(mol)]) == num_pairs
        assert len([*_get_14_pairs(mol.to_topology())]) == num_pairs

    def test_check_electrostatics_handlers(self, tip3p_missing_electrostatics_xml):
        # https://github.com/openforcefield/openff-toolkit/blob/0.10.2/openff/toolkit/data/test_forcefields/tip3p.offxml
        tip3p_missing_electrostatics = ForceField(tip3p_missing_electrostatics_xml)

        assert _check_electrostatics_handlers(tip3p_missing_electrostatics)

        tip3p_missing_electrostatics.deregister_parameter_handler("LibraryCharges")

        assert not _check_electrostatics_handlers(tip3p_missing_electrostatics)

    @pytest.mark.parametrize(
        ("smiles", "num_h_bonds"),
        [("C", 4), ("C#C", 2), ("O", 2)],
    )
    def test_get_num_h_bonds(self, smiles, num_h_bonds):
        topology = Molecule.from_smiles(smiles).to_topology()
        assert _get_num_h_bonds(topology) == num_h_bonds, smiles

    def test_combine_topologies(self):
        ethanol = Molecule.from_smiles("CCO")
        ethanol.name = "ETH"
        ethanol_topology = ethanol.to_topology()

        water = Molecule.from_smiles("O")
        water.name = "WAT"
        water_topology = water.to_topology()

        combined = _combine_topologies(ethanol_topology, water_topology)

        for attr in (
            "atoms",
            "bonds",
        ):
            attr = "n_" + attr
            assert getattr(combined, attr) == getattr(ethanol_topology, attr) + getattr(
                water_topology,
                attr,
            )
