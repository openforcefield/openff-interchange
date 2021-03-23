import pydantic
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff.parameters import AngleHandler, BondHandler
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.components.potentials import (
    Potential,
    PotentialHandler,
    WrappedPotential,
)
from openff.system.models import TopologyKey
from openff.system.tests.base_test import BaseTest


class TestWrappedPotential(BaseTest):
    def test_interpolated_potentials(self):
        """Test the construction of and .parameters getter of WrappedPotential"""

        bt = BondHandler.BondType(
            smirks="[#6X4:1]~[#8X2:2]",
            id="bbo1",
            k_bondorder1="100.0 * kilocalories_per_mole/angstrom**2",
            k_bondorder2="200.0 * kilocalories_per_mole/angstrom**2",
            length_bondorder1="1.4 * angstrom",
            length_bondorder2="1.3 * angstrom",
        )

        pot1 = Potential(
            parameters={"k": bt.k_bondorder[1], "length": bt.length_bondorder[1]}
        )
        pot2 = Potential(
            parameters={"k": bt.k_bondorder[2], "length": bt.length_bondorder[2]}
        )

        interp_pot = WrappedPotential(data={pot1: 0.2, pot2: 0.8})
        assert interp_pot.parameters == {
            "k": 180 * unit.Unit("kilocalorie / angstrom ** 2 / mole"),
            "length": 1.32 * unit.angstrom,
        }

        # Ensure a single Potential object can be wrapped with similar behavior
        simple = WrappedPotential(data=pot2)
        assert simple.parameters == pot2.parameters


class TestBondPotentialHandler(BaseTest):
    def test_dummy_potential_handler(self):
        handler = PotentialHandler(
            name="foo", expression="m*x+b", independent_variables="x"
        )
        assert handler.expression == "m*x+b"

        # Pydantic silently casts some types (int, float, Decimal) to str
        # in models that expect str; this test checks that the validator's
        # pre=True argument works;
        with pytest.raises(pydantic.ValidationError):
            PotentialHandler(name="foo", expression=1, independent_variables="x")

    def test_bond_potential_handler(self):
        top = Topology.from_molecules(Molecule.from_smiles("O=O"))

        bond_handler = BondHandler(version=0.3)
        bond_parameter = BondHandler.BondType(
            smirks="[*:1]~[*:2]",
            k=1.5 * omm_unit.kilocalorie_per_mole / omm_unit.angstrom ** 2,
            length=1.5 * omm_unit.angstrom,
            id="b1000",
        )
        bond_handler.add_parameter(bond_parameter.to_dict())

        from openff.system.stubs import ForceField

        forcefield = ForceField()
        forcefield.register_parameter_handler(bond_handler)
        bond_potentials = forcefield["Bonds"].create_potential(top)

        top_key = TopologyKey(atom_indices=(0, 1))
        pot = bond_potentials.potentials[bond_potentials.slot_map[top_key]]

        kcal_mol_a2 = unit.Unit("kilocalorie / (angstrom ** 2 * mole)")
        assert pot.parameters["k"].to(kcal_mol_a2).magnitude == pytest.approx(1.5)

    def test_angle_potential_handler(self):
        top = Topology.from_molecules(Molecule.from_smiles("CCC"))

        angle_handler = AngleHandler(version=0.3)
        angle_parameter = AngleHandler.AngleType(
            smirks="[*:1]~[*:2]~[*:3]",
            k=2.5 * omm_unit.kilocalorie_per_mole / omm_unit.radian ** 2,
            angle=100 * omm_unit.degree,
            id="b1000",
        )
        angle_handler.add_parameter(angle_parameter.to_dict())

        from openff.system.stubs import ForceField

        forcefield = ForceField()
        forcefield.register_parameter_handler(angle_handler)
        angle_potentials = forcefield["Angles"].create_potential(top)

        top_key = TopologyKey(atom_indices=(0, 1, 2))
        pot = angle_potentials.potentials[angle_potentials.slot_map[top_key]]

        kcal_mol_rad2 = unit.Unit("kilocalorie / (mole * radian ** 2)")
        assert pot.parameters["k"].to(kcal_mol_rad2).magnitude == pytest.approx(2.5)
