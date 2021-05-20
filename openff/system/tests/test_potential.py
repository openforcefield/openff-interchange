import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import AngleHandler, BondHandler
from openff.units import unit
from simtk import unit as omm_unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.potentials import PotentialHandler
from openff.system.components.smirnoff import SMIRNOFFAngleHandler, SMIRNOFFBondHandler
from openff.system.models import TopologyKey
from openff.system.tests.base_test import BaseTest


class TestBondPotentialHandler(BaseTest):
    def test_dummy_potential_handler(self):
        handler = PotentialHandler(
            type="foo",
            expression="m*x+b",
        )
        assert handler.type == "foo"
        assert handler.expression == "m*x+b"

    def test_bond_potential_handler(self):
        top = OFFBioTop.from_molecules(Molecule.from_smiles("O=O"))

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
        bond_potentials, _ = SMIRNOFFBondHandler.from_toolkit(
            bond_handler=forcefield["Bonds"],
            topology=top,
        )

        top_key = TopologyKey(atom_indices=(0, 1))
        pot = bond_potentials.potentials[bond_potentials.slot_map[top_key]]

        kcal_mol_a2 = unit.Unit("kilocalorie / (angstrom ** 2 * mole)")
        assert pot.parameters["k"].to(kcal_mol_a2).magnitude == pytest.approx(1.5)

    def test_angle_potential_handler(self):
        top = OFFBioTop.from_molecules(Molecule.from_smiles("CCC"))

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
        angle_potentials = SMIRNOFFAngleHandler.from_toolkit(
            parameter_handler=forcefield["Angles"],
            topology=top,
        )

        top_key = TopologyKey(atom_indices=(0, 1, 2))
        pot = angle_potentials.potentials[angle_potentials.slot_map[top_key]]

        kcal_mol_rad2 = unit.Unit("kilocalorie / (mole * radian ** 2)")
        assert pot.parameters["k"].to(kcal_mol_rad2).magnitude == pytest.approx(2.5)
