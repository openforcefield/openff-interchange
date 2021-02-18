import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path

from openff.system.exceptions import SMIRNOFFHandlersNotImplementedError
from openff.system.tests.base_test import BaseTest
from openff.system.tests.utils import compare_charges_omm_off, requires_pkg


class TestStubs(BaseTest):
    """Test the functionality of the stubs.py module"""

    def test_from_parsley(self, parsley):
        top = Topology.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = parsley.create_openff_system(top)

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()

    def test_unsupported_handler(self):
        from openff.system.stubs import ForceField

        gbsa_ff = ForceField(get_data_file_path("test_forcefields/GBSA_HCT-1.0.offxml"))

        with pytest.raises(
            SMIRNOFFHandlersNotImplementedError, match=r"SMIRNOFF parameters not.*GBSA"
        ):
            gbsa_ff.create_openff_system(topology=None)


class TestConstraints(BaseTest):
    """Test the handling of SMIRNOFF constraints"""

    def test_force_field_basic_constraints(self, parsley):
        """Test that a force field Constraints tag adds a
        Constraints handler with expected data"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        sys_out = parsley.create_openff_system(top)

        assert "Constraints" in sys_out.handlers.keys()
        constraints = sys_out.handlers["Constraints"]
        assert "(0, 1)" not in constraints.slot_map.keys()  # C-C bond
        assert "(0, 2)" in constraints.slot_map.keys()  # C-H bond
        assert len(constraints.slot_map.keys()) == 6  # number of C-H bonds
        assert len({constraints.slot_map.values()}) == 1  # always True
        assert (
            "distance"
            in constraints.constraints[constraints.slot_map["(0, 2)"]].parameters
        )

    def test_force_field_no_constraints(self, parsley_unconstrained):
        """Test that a force field _without_ a Constraints tag does not add a
        Constraints handler"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        sys_out = parsley_unconstrained.create_openff_system(top)

        assert "Constraints" not in sys_out.handlers.keys()

    def test_no_matched_constraints(self, parsley):
        """Test that a force field with a Constraints tag adds an empty
        Constraints handler when the topology does not match any parameters"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("O=C=O"))
        sys_out = parsley.create_openff_system(top)

        assert "Constraints" in sys_out.handlers.keys()

        constraints = sys_out.handlers["Constraints"]
        assert constraints.slot_map == dict()
        assert constraints.constraints == dict()

    def test_constraint_reassignment(self, parsley):
        """Test that constraints already existing in a parametrized system
        can be updated against new force field data"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CCO"))
        constrained = parsley.create_openff_system(top)

        assert len(constrained.handlers["Constraints"].slot_map.keys()) == 6

        # Update Parsley to also constrain C-O bonds
        parsley["Constraints"].add_parameter({"smirks": "[#6:1]-[#8:2]"})

        constrained.handlers["Constraints"].store_matches(
            parameter_handler=parsley["Constraints"],
            topology=top,
        )

        from openff.system.components.smirnoff import SMIRNOFFBondHandler

        bond_handler = SMIRNOFFBondHandler()
        bond_handler.store_matches(parameter_handler=parsley["Bonds"], topology=top)
        bond_handler.store_potentials(parameter_handler=parsley["Bonds"])

        constrained.handlers["Constraints"].store_constraints(
            parameter_handler=parsley["Constraints"],
            bond_handler=bond_handler,
        )

        assert len(constrained.handlers["Constraints"].slot_map.keys()) == 7


class TestChargeAssignment(BaseTest):
    def test_default_am1bcc_charge_assignment(self, parsley):
        top = Topology.from_molecules(
            [
                Molecule.from_smiles("C"),
                Molecule.from_smiles("C=C"),
                Molecule.from_smiles("CCO"),
            ]
        )

        reference = parsley.create_openmm_system(top)
        new = parsley.create_openff_system(top)

        compare_charges_omm_off(reference, new)

    @requires_pkg("openff.recharge")
    def test_charge_increment_assignment(self, parsley):
        from openff.recharge.charges.bcc import original_am1bcc_corrections
        from openff.recharge.smirnoff import to_smirnoff

        top = Topology.from_molecules(
            [
                Molecule.from_smiles("C"),
                Molecule.from_smiles("C=C"),
                Molecule.from_smiles("CCO"),
            ]
        )

        recharge_bccs = to_smirnoff(original_am1bcc_corrections())
        recharge_bccs.partial_charge_method = "AM1-Mulliken"

        parsley.deregister_parameter_handler("ToolkitAM1BCC")
        parsley.register_parameter_handler(recharge_bccs)

        reference = parsley.create_openmm_system(top)
        new = parsley.create_openff_system(top)

        compare_charges_omm_off(reference, new)

    def test_library_charge_assignment(self):
        from openff.system.stubs import ForceField

        forcefield = ForceField("openff-1.3.0.offxml")
        forcefield.deregister_parameter_handler("ToolkitAM1BCC")

        top = Topology.from_molecules(
            [Molecule.from_smiles(smi) for smi in ["[Na+]", "[Cl-]"]]
        )

        reference = forcefield.create_openmm_system(top)
        new = forcefield.create_openff_system(top)

        compare_charges_omm_off(reference, new)
