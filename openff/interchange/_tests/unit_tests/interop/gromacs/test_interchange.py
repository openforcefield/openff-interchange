import pytest
from openff.toolkit import Molecule, Topology
from openff.units import unit

from openff.interchange._tests import _BaseTest
from openff.interchange.interop.gromacs._interchange import _convert_topology
from openff.interchange.interop.gromacs.models.models import GROMACSSystem
from openff.interchange.smirnoff._gromacs import _convert


class TestConvertTopology(_BaseTest):
    @pytest.fixture()
    def simple_system(self, sage_unconstrained) -> GROMACSSystem:
        molecule = Molecule.from_smiles("CCO")
        molecule.generate_conformers(n_conformers=1)
        topology = Topology.from_molecules([molecule])
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        return _convert(sage_unconstrained.create_interchange(topology))

    @pytest.fixture()
    def water_dimer(self, sage_unconstrained):
        water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
        water.name = "WAT"
        water.generate_conformers(n_conformers=1)
        topology = Topology.from_molecules([water, water])
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        return _convert(sage_unconstrained.create_interchange(topology))

    def test_convert_basic_system(self, simple_system):
        converted = _convert_topology(simple_system)

        assert converted.n_molecules == 1
        assert converted.n_atoms == 9
        assert converted.n_bonds == 8

    def test_water_with_settles_has_bonds_in_topology(self, water_dimer):
        assert "WAT" in water_dimer.molecule_types
        assert len(water_dimer.molecule_types["WAT"].settles) == 1

        converted = _convert_topology(water_dimer)

        assert converted.molecule(0).name == "WAT"
        assert converted.molecule(1).name == "WAT"
        assert converted.n_molecules == 2
        assert converted.n_atoms == 6
        assert converted.n_bonds == 4

        assert [atom.atomic_number for atom in converted.atoms] == [8, 1, 1, 8, 1, 1]
