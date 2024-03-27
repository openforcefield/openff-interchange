import pytest
from openff.toolkit import Molecule, Topology, unit
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.utilities.testing import skip_if_missing

from openff.interchange._tests import get_test_file_path
from openff.interchange.components.toolkit import (
    _check_electrostatics_handlers,
    _combine_topologies,
    _get_14_pairs,
    _get_num_h_bonds,
    _lookup_virtual_site_parameter,
    _simple_topology_from_openmm,
)


@pytest.fixture()
def simple_methane():
    return _SimpleMolecule.from_molecule(Molecule.from_smiles("C"))


@pytest.fixture()
def simple_water(water):
    return _SimpleMolecule.from_molecule(water)


def test_simple_topology_uniqueness(simple_methane, simple_water):
    topology = Topology.from_molecules(
        [
            simple_methane,
            simple_water,
            simple_methane,
            simple_methane,
            simple_water,
        ],
    )
    assert len(topology.identical_molecule_groups) == 2


class TestToolkitUtils:
    @pytest.mark.parametrize(
        ("smiles", "num_pairs"),
        [
            ("C#C", 1),
            ("CCO", 12),
            ("C1=CC=CC=C1", 24),
            ("C=1=C=C1", 0),
            ("C=1=C=C=C1", 0),
            ("C=1(Cl)-C(Cl)=C1", 1),
            ("C=1=C(Cl)C(=C=1)Cl", 5),
        ],
    )
    def test_get_14_pairs(self, smiles, num_pairs):
        mol = Molecule.from_smiles(smiles)
        assert len([*_get_14_pairs(mol)]) == num_pairs
        assert len([*_get_14_pairs(mol.to_topology())]) == num_pairs

    def test_check_electrostatics_handlers(self, tip3p):
        tip3p.deregister_parameter_handler("Electrostatics")

        assert _check_electrostatics_handlers(tip3p)

        tip3p.deregister_parameter_handler("LibraryCharges")

        assert not _check_electrostatics_handlers(tip3p)

    @pytest.mark.parametrize(
        ("smiles", "num_h_bonds"),
        [("C", 4), ("C#C", 2), ("O", 2)],
    )
    def test_get_num_h_bonds(self, smiles, num_h_bonds):
        topology = Molecule.from_smiles(smiles).to_topology()
        assert _get_num_h_bonds(topology) == num_h_bonds, smiles

    def test_combine_topologies(self, water):
        ethanol = Molecule.from_smiles("CCO")
        ethanol.name = "ETH"
        ethanol_topology = ethanol.to_topology()

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

    @skip_if_missing("openmm")
    def test_simple_topology_from_openmm(self):
        simple_topology = _simple_topology_from_openmm(
            Topology.from_molecules(
                [
                    Molecule.from_smiles("O"),
                    Molecule.from_smiles("CCO"),
                ],
            ).to_openmm(),
        )

        assert all(
            isinstance(molecule, _SimpleMolecule)
            for molecule in simple_topology.molecules
        )

        assert sorted(molecule.n_atoms for molecule in simple_topology.molecules) == [
            3,
            9,
        ]

    def test_simple_subgraph_atom_ordering(self):
        """
        Test that simple molecules created from subgraphs use nodes in ascending
        order (like you'd always do for atom indices)
        """
        pytest.importorskip("openmm")

        import openmm.app

        crazy_water = openmm.app.PDBFile(
            get_test_file_path("copied_reordered_water.pdb").as_posix(),
        )

        expected_atomic_numbers = [
            8,
            1,
            1,
            1,
            8,
            1,
            1,
            1,
            8,
            1,
            8,
            1,
        ]

        simple_topology = _simple_topology_from_openmm(crazy_water.topology)

        for atom, expected_atomic_number in zip(
            simple_topology.atoms,
            expected_atomic_numbers,
        ):
            assert atom.atomic_number == expected_atomic_number


def test_lookup_virtual_site_parameter(
    sage,
    tip4p,
    sage_with_two_virtual_sites_same_smirks,
):
    with pytest.raises(
        NotImplementedError,
    ):
        _lookup_virtual_site_parameter(
            sage["LibraryCharges"],
            smirks="",
            name="",
            match="",
        )

    assert _lookup_virtual_site_parameter(
        parameter_handler=tip4p["VirtualSites"],
        smirks="[#1:2]-[#8X2H2+0:1]-[#1:3]",
        name="EP",
        match="once",
    )

    ep1 = _lookup_virtual_site_parameter(
        sage_with_two_virtual_sites_same_smirks["VirtualSites"],
        smirks="[#7:1](-[*:2])(-[*:3])-[*:4]",
        name="EP1",
        match="once",
    )
    assert ep1.distance.m_as(unit.nanometer) == -0.5

    ep2 = _lookup_virtual_site_parameter(
        sage_with_two_virtual_sites_same_smirks["VirtualSites"],
        smirks="[#7:1](-[*:2])(-[*:3])-[*:4]",
        name="EP2",
        match="once",
    )
    assert ep2.distance.m_as(unit.nanometer) == 1.5

    with pytest.raises(
        ValueError,
        match="No VirtualSiteType found with .*EP3",
    ):
        _lookup_virtual_site_parameter(
            sage_with_two_virtual_sites_same_smirks["VirtualSites"],
            smirks="[#7:1](-[*:2])(-[*:3])-[*:4]",
            name="EP3",
            match="once",
        )
