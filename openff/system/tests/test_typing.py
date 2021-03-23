import numpy as np
import pytest
from openff.toolkit.topology import Molecule

from openff.system import unit
from openff.system.exceptions import MissingBondOrdersError
from openff.system.models import TopologyKey
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):
    # There's probably a better way to this, but pytest doesn't let fixtures be passed to parametrize
    # TODO: check for proper conversion, not just completeness
    def test_reconstruct_toolkit_forcefield(
        self,
        argon_ff,
        argon_top,
        ammonia_ff,
        ammonia_top,
    ):

        argon_sys = argon_ff.create_openff_system(argon_top)
        assert "vdW" in argon_sys.handlers
        assert "Electrostatics" in argon_sys.handlers
        assert "LibraryCharges" in argon_sys.handlers

        vdw_handler = argon_sys["vdW"]
        found_smirks = [key.id for key in vdw_handler.slot_map.values()]
        assert all([smirks == "[#18:1]" for smirks in found_smirks])

        ammonia_sys = ammonia_ff.create_openff_system(ammonia_top)

        expected = ["Angles", "Bonds", "vdW"]
        assert sorted(ammonia_sys.handlers.keys()) == sorted(expected)

        for handler in ammonia_sys.handlers.values():
            assert len([*handler.potentials.items()]) > 0


class TestParameterInterpolation(BaseTest):
    xml_ff_bo = """<?xml version='1.0' encoding='ASCII'?>
    <SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
      <Bonds version="0.3" fractional_bondorder_method="AM1-Wiberg"
        fractional_bondorder_interpolation="linear">
        <Bond
          smirks="[#6X4:1]~[#8X2:2]"
          id="bbo1"
          k_bondorder1="100.0 * kilocalories_per_mole/angstrom**2"
          k_bondorder2="500.0 * kilocalories_per_mole/angstrom**2"
          length_bondorder1="1.4 * angstrom"
          length_bondorder2="1.3 * angstrom"
          />
      </Bonds>
    </SMIRNOFF>
    """

    def test_bond_order_interpolation(self):
        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml", self.xml_ff_bo
        )

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)

        with pytest.raises(MissingBondOrdersError):
            forcefield.create_openff_system(mol.to_topology())

        mol.assign_fractional_bond_orders()
        mol.bonds[1].fractional_bond_order = 1.5

        out = forcefield.create_openff_system(mol.to_topology())

        assert out["Bonds"].potentials[
            out["Bonds"].slot_map[TopologyKey(atom_indices=(1, 2))]
        ].parameters["k"] == 300 * unit.Unit("kilocalories / mol / angstrom ** 2")

    def test_bond_order_interpolation_similar_bonds(self):
        """Test that key mappings do not get confused when two bonds having similar SMIRKS matches
        have different bond orders"""
        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml", self.xml_ff_bo
        )

        # TODO: Construct manually to avoid relying on atom ordering
        mol = Molecule.from_smiles("C(CCO)O")
        mol.generate_conformers(n_conformers=1)

        mol.bonds[2].fractional_bond_order = 1.5
        mol.bonds[3].fractional_bond_order = 1.2

        out = forcefield.create_openff_system(mol.to_topology())

        bond1_top_key = TopologyKey(atom_indices=(2, 3))
        bond1_pot_key = out["Bonds"].slot_map[bond1_top_key]

        bond2_top_key = TopologyKey(atom_indices=(0, 4))
        bond2_pot_key = out["Bonds"].slot_map[bond2_top_key]

        assert np.allclose(
            out["Bonds"].potentials[bond1_pot_key].parameters["k"],
            300.0 * unit.Unit("kilocalories / mol / angstrom ** 2"),
        )

        assert np.allclose(
            out["Bonds"].potentials[bond2_pot_key].parameters["k"],
            180.0 * unit.Unit("kilocalories / mol / angstrom ** 2"),
        )


#    def test_more_map_functions(self, parsley, cyclohexane_top):
#        # TODO: Better way of testing individual handlers
#
#        term_collection = SMIRNOFFTermCollection()
#
#        SUPPORTED_HANDLER_MAPPING.pop("Electrostatics")
#
#        # TODO: This should just be
#        # term_collection = SMIRNOFFTermCollection.from_toolkit_data(parsley, cyclohexane_top)
#        for name, handler in parsley._parameter_handlers.items():
#            if name in SUPPORTED_HANDLER_MAPPING.keys():
#                term_collection.add_parameter_handler(
#                    handler=handler, topology=cyclohexane_top, forcefield=parsley
#                )
#
#        # Do this in separate steps so that Electrostatics handler can access the handlers it depends on
#        handlers_to_drop = []
#        for name in parsley._parameter_handlers.keys():
#            if name not in SUPPORTED_HANDLER_MAPPING.keys():
#                handlers_to_drop.append(name)
#
#        for name in handlers_to_drop:
#            parsley._parameter_handlers.pop(name)
#
#        found_keys = sorted(term_collection.terms.keys())
#        expected_keys = sorted(SUPPORTED_HANDLER_MAPPING.keys())
#        assert found_keys == expected_keys
#
#    def test_construct_term_from_toolkit_forcefield(self, parsley, ethanol_top):
#        SMIRNOFFvdWTerm.build_from_toolkit_data(
#            handler=parsley["vdW"],
#            topology=ethanol_top,
#        )
#
#        ref = get_partial_charges_from_openmm_system(
#            parsley.create_openmm_system(ethanol_top)
#        )
#
#        electrostatics_term = ElectrostaticsTerm.build_from_toolkit_data(
#            forcefield=parsley,
#            topology=ethanol_top,
#        )
#        partial_charges = unwrap_list_of_pint_quantities(
#            [*electrostatics_term.potentials.values()]
#        )
#
#        assert np.allclose(partial_charges, ref)
#
#    @pytest.mark.skip
#    def test_unimplemented_conversions(self, parsley, ethanol_top):
#
#        # TODO: Replace this with a system contained a truly unsupported potential
#        # with pytest.raises(SMIRNOFFHandlerNotImplementedError):
#        SMIRNOFFTermCollection.from_toolkit_data(parsley, ethanol_top)
#
#
# class TestSMIRNOFFTerms(BaseTest):
#    handler_expression_mapping = {
#        "vdW": "4*epsilon*((sigma/r)**12-(sigma/r)**6)",
#        "Bonds": "1/2*k*(length-length_0)**2",
#        "Angles": "1/2*k*(angle-angle_0)**2",
#    }
#
#    @pytest.mark.parametrize(
#        "handler_name,expression",
#        [
#            ("vdW", "4*epsilon*((sigma/r)**12-(sigma/r)**6)"),
#            ("Bonds", "1/2*k*(length-length_0)**2"),
#            ("Angles", "1/2*k*(angle-angle_0)**2"),
#        ],
#    )
#    def test_smirnoff_terms(self, parsley, ethanol_top, handler_name, expression):
#        smirnoff_term = SMIRNOFFPotentialTerm.build_from_toolkit_data(
#            handler=parsley[handler_name],
#            topology=ethanol_top,
#            forcefield=None,
#        )
#
#        assert smirnoff_term.smirks_map == build_slot_smirks_map_term(
#            parsley[handler_name],
#            ethanol_top,
#        )
#
#        assert smirnoff_term.potentials == build_smirks_potential_map_term(
#            handler=parsley[handler_name],
#            smirks_map=smirnoff_term.smirks_map,
#        )
#
#        for smirks, pot in smirnoff_term.potentials.items():
#            assert pot.expression == expression
#
#
# class TestSMIRNOFFvdWTerm(BaseTest):
#    def test_basic_constructor(self, ethanol_top, parsley):
#        SMIRNOFFvdWTerm.build_from_toolkit_data(parsley["vdW"], ethanol_top)
#
#    def test_scaling_factors(self, ethanol_top, parsley):
#        parsley_vdw = parsley["vdW"]
#        ref = SMIRNOFFvdWTerm.build_from_toolkit_data(parsley_vdw, ethanol_top)
#        factors = [ref.scale12, ref.scale13, ref.scale14, ref.scale15]
#
#        assert factors == [0.0, 0.0, 0.5, 1.0]
#
#        # Cannot handle non-zero scale12 or scale13 with current toolkit
#        parsley_vdw.scale14 = 14.14
#
#        mod = SMIRNOFFvdWTerm.build_from_toolkit_data(parsley_vdw, ethanol_top)
#        factors = [mod.scale12, mod.scale13, mod.scale14, mod.scale15]
#
#        assert factors == [0.0, 0.0, 14.14, 1.0]
#
#
# class TestElectrostaticsTerm(BaseTest):
#    def test_build_dummy_electrostatics(self, argon_ff, argon_top):
#        ElectrostaticsTerm.build_from_toolkit_data(
#            forcefield=argon_ff,
#            topology=argon_top,
#        )
#
#    def test_build_parsley_electrostatics(self, parsley, ethanol_top):
#        ElectrostaticsTerm.build_from_toolkit_data(
#            forcefield=parsley,
#            topology=ethanol_top,
#        )
#
