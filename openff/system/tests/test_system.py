import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.exceptions import (
    InvalidTopologyError,
    SMIRNOFFHandlersNotImplementedError,
)
from openff.system.models import TopologyKey
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.utils import compare_charges_omm_off


def test_getitem():
    """Test behavior of System.__getitem__"""
    mol = Molecule.from_smiles("CCO")
    parsley = ForceField("openff-1.0.0.offxml")

    out = System.from_smirnoff(parsley, mol.to_topology())
    out.box = [4, 4, 4]

    assert not out.positions
    np.testing.assert_equal(out["box"].m, (4 * np.eye(3) * unit.nanometer).m)
    np.testing.assert_equal(out["box"].m, out["box_vectors"].m)

    assert out["Bonds"] == out.handlers["Bonds"]

    with pytest.raises(LookupError, match="Only str"):
        out[1]

    with pytest.raises(LookupError, match="Could not find"):
        out["CMAPs"]


class TestFromSMIRNOFF(BaseTest):
    """Test the functionality of the stubs.py module"""

    def test_from_parsley(self, parsley):
        top = OFFBioTop.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = System.from_smirnoff(parsley, top)

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()

        assert type(out.topology) == OFFBioTop
        assert type(out.topology) != Topology
        assert isinstance(out.topology, Topology)

    def test_unsupported_handler(self):
        gbsa_ff = ForceField(get_data_file_path("test_forcefields/GBSA_HCT-1.0.offxml"))

        with pytest.raises(
            SMIRNOFFHandlersNotImplementedError, match=r"SMIRNOFF parameters not.*GBSA"
        ):
            System.from_smirnoff(gbsa_ff, topology=None)

    def test_unsupported_topology(self, parsley, ethanol_top):
        mdtop = md.Topology.from_openmm(ethanol_top.to_openmm())

        with pytest.raises(InvalidTopologyError, match="mdtraj.core.*Topology"):
            System.from_smirnoff(parsley, mdtop)

        with pytest.raises(InvalidTopologyError, match="simtk.*app.*Topology"):
            System.from_smirnoff(parsley, mdtop)

    def test_force_field_basic_constraints(self, parsley):
        """Test that a force field Constraints tag adds a
        Constraints handler with expected data"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        sys_out = System.from_smirnoff(parsley, top)

        assert "Constraints" in sys_out.handlers.keys()
        constraints = sys_out.handlers["Constraints"]
        c_c_bond = TopologyKey(atom_indices=(0, 1))  # C-C bond
        assert c_c_bond not in constraints.slot_map.keys()
        c_h_bond = TopologyKey(atom_indices=(0, 2))  # C-H bond
        assert c_h_bond in constraints.slot_map.keys()
        assert len(constraints.slot_map.keys()) == 6  # number of C-H bonds
        assert len({constraints.slot_map.values()}) == 1  # always True
        assert (
            "distance"
            in constraints.constraints[constraints.slot_map[c_h_bond]].parameters
        )

    def test_force_field_no_constraints(self, parsley_unconstrained):
        """Test that a force field _without_ a Constraints tag does not add a
        Constraints handler"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        sys_out = System.from_smirnoff(parsley_unconstrained, top)

        assert "Constraints" not in sys_out.handlers.keys()

    def test_no_matched_constraints(self, parsley):
        """Test that a force field with a Constraints tag adds an empty
        Constraints handler when the topology does not match any parameters"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("O=C=O"))
        sys_out = System.from_smirnoff(parsley, top)

        assert "Constraints" in sys_out.handlers.keys()

        constraints = sys_out.handlers["Constraints"]
        assert constraints.slot_map == dict()
        assert constraints.constraints == dict()

    @pytest.mark.skip(reason="Needs to be updated to be a part of SMIRNOFFBondHandler")
    def test_constraint_reassignment(self, parsley):
        """Test that constraints already existing in a parametrized system
        can be updated against new force field data"""
        # TODO: Replace with more minimal force field

        top = Topology.from_molecules(Molecule.from_smiles("CCO"))
        constrained = System.from_smirnoff(parsley, top)

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

    @pytest.mark.slow
    def test_default_am1bcc_charge_assignment(self, parsley):
        top = Topology.from_molecules(
            [
                Molecule.from_smiles("C"),
                Molecule.from_smiles("C=C"),
                Molecule.from_smiles("CCO"),
            ]
        )

        reference = parsley.create_openmm_system(top)

        new = System.from_smirnoff(parsley, top)

        compare_charges_omm_off(reference, new)

    @skip_if_missing("openff.recharge")
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
        new = System.from_smirnoff(parsley, top)

        compare_charges_omm_off(reference, new)

    def test_library_charge_assignment(self):
        forcefield = ForceField("openff-1.3.0.offxml")
        forcefield.deregister_parameter_handler("ToolkitAM1BCC")

        top = Topology.from_molecules(
            [Molecule.from_smiles(smi) for smi in ["[Na+]", "[Cl-]"]]
        )

        reference = forcefield.create_openmm_system(top)
        new = System.from_smirnoff(forcefield, top)

        compare_charges_omm_off(reference, new)
