import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, RDKitToolkitWrapper, Topology, unit
from openff.toolkit.typing.engines.smirnoff import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    NAGLChargesHandler,
    ToolkitAM1BCCHandler,
    vdWHandler,
)
from openff.toolkit.utils.exceptions import MissingPackageError, SMIRNOFFVersionError
from openff.toolkit.utils.toolkit_registry import ToolkitRegistry, toolkit_registry_manager
from packaging.version import Version

from openff.interchange import Interchange
from openff.interchange.exceptions import NonIntegralMoleculeChargeError
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    _downconvert_vdw_handler,
    _upconvert_vdw_handler,
)


class TestNonbonded:
    def test_electrostatics_am1_handler(self, methane):
        methane.assign_partial_charges(partial_charge_method="am1bcc")

        # Explicitly store these, since results differ RDKit/AmberTools vs. OpenEye
        reference_charges = [c.m for c in methane.partial_charges]

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            ToolkitAM1BCCHandler(version=0.3),
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsCollection.create(
            parameter_handlers,
            methane.to_topology(),
        )
        numpy.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler._get_charges().values()],
            reference_charges,
        )

    def test_electrostatics_library_charges(self, methane):
        library_charge_handler = LibraryChargeHandler(version=0.3)
        library_charge_handler.add_parameter(
            {
                "smirks": "[#6X4:1]-[#1:2]",
                "charge1": -0.1 * unit.elementary_charge,
                "charge2": 0.025 * unit.elementary_charge,
            },
        )

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            library_charge_handler,
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsCollection.create(
            parameter_handlers,
            methane.to_topology(),
        )

        numpy.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler._get_charges().values()],
            [-0.1, 0.025, 0.025, 0.025, 0.025],
        )

    def test_electrostatics_charge_increments(self, hydrogen_chloride):
        hydrogen_chloride.assign_partial_charges(partial_charge_method="am1-mulliken")

        reference_charges = [c.m for c in hydrogen_chloride.partial_charges]
        reference_charges[0] += 0.1
        reference_charges[1] -= 0.1

        charge_increment_handler = ChargeIncrementModelHandler(version=0.3)
        charge_increment_handler.add_parameter(
            {
                "smirks": "[#17:1]-[#1:2]",
                "charge_increment1": 0.1 * unit.elementary_charge,
                "charge_increment2": -0.1 * unit.elementary_charge,
            },
        )

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            charge_increment_handler,
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsCollection.create(
            parameter_handlers,
            hydrogen_chloride.to_topology(),
        )

        # AM1-Mulliken charges are [-0.168,  0.168], increments are [0.1, -0.1],
        # sum is [-0.068,  0.068]
        numpy.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler._get_charges().values()],
            reference_charges,
        )

    @pytest.mark.slow
    def test_toolkit_am1bcc_uses_elf10_if_oe_is_available(self, sage, hexane_diol):
        """
        Ensure that the ToolkitAM1BCCHandler assigns ELF10 charges if OpenEye is available.

        Taken from https://github.com/openforcefield/openff-toolkit/pull/1214,
        """
        try:
            hexane_diol.assign_partial_charges(partial_charge_method="am1bccelf10")
            uses_elf10 = True
        except ValueError:
            # This assumes that the ValueError stems from "am1bccelf10" and not other sources; the
            # toolkit should implement a failure mode that does not clash with other `ValueError`s
            hexane_diol.assign_partial_charges(partial_charge_method="am1bcc")
            uses_elf10 = False

        partial_charges = [c.m for c in hexane_diol.partial_charges]

        assigned_charges = [
            v.m for v in Interchange.from_smirnoff(sage, [hexane_diol])["Electrostatics"]._get_charges().values()
        ]

        try:
            from openeye import oechem

            openeye_available = oechem.OEChemIsLicensed()
        except ImportError:
            openeye_available = False

        if openeye_available:
            assert uses_elf10
            numpy.testing.assert_allclose(partial_charges, assigned_charges)
        else:
            assert not uses_elf10
            numpy.testing.assert_allclose(partial_charges, assigned_charges)

    def test_nagl_charge_assignment_matches_reference(self, sage_nagl, hexane_diol):
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        # Leave the ToolkitAM1BCC tag in openff-2.1.0 to ensure that the NAGLCharges handler takes precedence

        interchange = sage_nagl.create_interchange(topology=hexane_diol.to_topology())

        assigned_charges_unitless = interchange["Electrostatics"].get_charge_array().m

        expected_charges = hexane_diol.partial_charges
        assert expected_charges is not None
        assert expected_charges.units == unit.elementary_charge
        assert not all(charge == 0 for charge in expected_charges.magnitude)
        expected_charges_unitless = [v.m for v in expected_charges]
        numpy.testing.assert_allclose(expected_charges_unitless, assigned_charges_unitless)


class TestNAGLChargesErrorHandling:
    """Test NAGLCharges error conditions."""

    def test_nagl_charges_missing_toolkit_error(self, sage_nagl, hexane_diol):
        """Test MissingPackageError when NAGL toolkit is not available. This should fail immediately instead of falling
        back to ToolkitAM1BCC, since it doesn't know whether the molecule would have successfully
        had charges assigned by NAGL if it were available."""

        # Mock the toolkit registry to not have NAGL
        # RDKit is needed for SMARTS matching.
        with toolkit_registry_manager(ToolkitRegistry(toolkit_precedence=[RDKitToolkitWrapper])):
            with pytest.raises(MissingPackageError, match="NAGL software isn't present"):
                sage_nagl.create_interchange(topology=hexane_diol.to_topology())

            # No error should be raised if using charge_from_molecules
            sage_nagl.create_interchange(
                topology=hexane_diol.to_topology(),
                charge_from_molecules=[hexane_diol],
            )

    def test_nagl_charges_invalid_model_file(self, sage, hexane_diol):
        """Test error handling for invalid model file paths. This should fail immediately instead of falling
        back to ToolkitAM1BCC, since it doesn't know whether the molecule would have successfully
        had charges assigned by this model it had been able to find it."""
        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "nonexistent_model.pt",
                "version": "0.3",
            },
        )
        with pytest.raises(FileNotFoundError):
            sage.create_interchange(topology=hexane_diol.to_topology())

        sage["NAGLCharges"].model_file = ""
        with pytest.raises(FileNotFoundError):
            sage.create_interchange(topology=hexane_diol.to_topology())

        sage["NAGLCharges"].model_file = None
        with pytest.raises(FileNotFoundError):
            sage.create_interchange(topology=hexane_diol.to_topology())

    def test_nagl_charges_bad_hash(self, sage, hexane_diol, monkeypatch):
        """Test error handling for a bad hash. This should fail immediately instead of falling
        back to ToolkitAM1BCC, since it doesn't know whether the molecule would have successfully
        had charges assigned by this model if the hash comparison hadn't failed."""
        from openff.nagl_models._dynamic_fetch import HashComparisonFailedException

        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
                "model_file_hash": "bad_hash",
                "version": "0.3",
            },
        )
        with pytest.raises(HashComparisonFailedException):
            sage.create_interchange(topology=hexane_diol.to_topology())

    def test_nagl_charges_bad_doi(self, sage, hexane_diol, monkeypatch):
        """Test error handling for a bad DOI. This should fail immediately instead of falling
        back to ToolkitAM1BCC, since it doesn't know whether the molecule would have successfully
        had charges assigned by this model, since it's unfetchable."""
        from openff.nagl_models._dynamic_fetch import UnableToParseDOIException

        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "nonexistent_model.pt",
                "digital_object_identifier": "blah.foo/bar",
                "version": "0.3",
            },
        )
        with pytest.raises(UnableToParseDOIException):
            sage.create_interchange(topology=hexane_diol.to_topology())

    # For more information on why this test is skipped, see
    # https://github.com/openforcefield/openff-interchange/pull/1206/commits/f99a10e17ad56235ba1f36ae35f6383a22ed840a#r2248864028
    @pytest.mark.xfail(
        reason="charge assignment handler fallback behavior not yet implemented",
        raises=ValueError,
    )
    def test_nagl_charges_fallback_to_charge_increment_model(self, sage):
        """Test that NAGL falls back to ChargeIncrementModel when molecule contains unsupported elements."""
        pytest.importorskip("openff.nagl")

        # Create a boron-containing molecule with nonzero formal charge
        # BF4- anion - boron is not supported by current NAGL models
        boron_molecule = Molecule.from_smiles("[B-]([F])([F])([F])[F]")

        # Verify formal charges are not all zero
        formal_charges = [atom.formal_charge.m for atom in boron_molecule.atoms]
        assert not all(charge == 0 for charge in formal_charges)

        # Create minimal force field with only the needed handlers
        ff = ForceField()

        # Add Electrostatics handler
        ff.register_parameter_handler(
            ElectrostaticsHandler(version="0.4"),
        )

        # Add NAGLCharges handler
        ff.register_parameter_handler(
            NAGLChargesHandler(
                version="0.3",
                model_file="openff-gnn-am1bcc-0.1.0-rc.3.pt",
            ),
        )

        # Add ChargeIncrementModel handler with formal_charge method and no increments
        charge_increment_handler = ChargeIncrementModelHandler(
            version="0.3",
            partial_charge_method="formal_charge",
        )
        ff.register_parameter_handler(charge_increment_handler)

        # Should succeed despite NAGL not supporting boron
        interchange = ff.create_interchange(topology=boron_molecule.to_topology())

        # Should have assigned charges to all atoms
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Assigned charges should match formal charges (fallback to ChargeIncrementModel)
        expected_charges = [atom.formal_charge.m for atom in boron_molecule.atoms]
        numpy.testing.assert_allclose(assigned_charges.m, expected_charges)

        # Net charge should match molecule's total formal charge
        assert abs(sum(assigned_charges.m) - boron_molecule.total_charge.m) < 1e-10

    @pytest.mark.xfail(
        reason="charge assignment handler fallback behavior not yet implemented",
        raises=ValueError,
    )
    def test_nagl_charges_all_handlers_fail_comprehensive_error(self, sage):
        """Test error reporting when all charge assignment methods fail."""
        pytest.importorskip("openff.nagl")

        # Create a uranium compound - not supported by any current charge assignment method
        uranium_molecule = Molecule.from_smiles("[U+4]")

        # Create force field with multiple charge assignment handlers
        ff = ForceField()

        # Add Electrostatics handler
        ff.register_parameter_handler(
            ElectrostaticsHandler(version="0.4"),
        )

        # Add NAGLCharges handler
        ff.register_parameter_handler(
            NAGLChargesHandler(
                version="0.3",
                model_file="openff-gnn-am1bcc-0.1.0-rc.3.pt",
            ),
        )

        # Add ToolkitAM1BCC handler
        ff.register_parameter_handler(
            ToolkitAM1BCCHandler(version="0.3"),
        )

        # Add ChargeIncrementModel handler with gasteiger method
        charge_increment_handler = ChargeIncrementModelHandler(
            version="0.3",
            partial_charge_method="mmff94",
        )
        ff.register_parameter_handler(charge_increment_handler)

        # Should fail with comprehensive error message
        with pytest.raises(RuntimeError) as excinfo:
            ff.create_interchange(topology=uranium_molecule.to_topology())

        error_message = str(excinfo.value)

        # Error should mention that no charges could be assigned
        assert "could not be fully assigned charges" in error_message

        # Error should contain information about each handler's failure
        assert "NAGLCharges" in error_message
        assert "ToolkitAM1BCC" in error_message
        assert "ChargeIncrementModel" in error_message

        # Should mention the exceptions raised by various handlers
        assert "exceptions raised by various handlers" in error_message


class TestNAGLChargesPrecedence:
    """Test NAGLCharges precedence in the hierarchy of charge assignment methods."""

    def test_nagl_charges_precedence_over_am1bcc(self, sage_nagl, hexane_diol):
        """Test that NAGLCharges takes precedence over ToolkitAM1BCC."""
        sage_nagl.get_parameter_handler("ToolkitAM1BCC", {"version": "0.3"})
        # Get reference charges from NAGL
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_charges = [c.m for c in hexane_diol.partial_charges]

        # Get reference charges from AM1BCC
        hexane_diol.assign_partial_charges("am1bcc")
        am1bcc_charges = [c.m for c in hexane_diol.partial_charges]

        # Ensure they're different
        assert not numpy.allclose(nagl_charges, am1bcc_charges)

        interchange = sage_nagl.create_interchange(topology=hexane_diol.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match NAGL charges, not AM1BCC
        numpy.testing.assert_allclose(assigned_charges, nagl_charges)

    def test_library_charges_precedence_over_nagl(self, sage_nagl, methane):
        """Test that LibraryCharges takes precedence over NAGLCharges."""

        sage_nagl["LibraryCharges"].add_parameter(
            {
                "smirks": "[#6X4:1]-[#1:2]",
                "charge1": -0.2 * unit.elementary_charge,
                "charge2": 0.05 * unit.elementary_charge,
            },
        )

        interchange = sage_nagl.create_interchange(topology=methane.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match library charges
        expected_charges = [-0.2, 0.05, 0.05, 0.05, 0.05]
        numpy.testing.assert_allclose(assigned_charges, expected_charges)

    def test_nagl_charges_precedence_over_charge_increments(self, sage_nagl, hexane_diol):
        """Test that NAGLCharges takes precedence over ChargeIncrementModel as base charges."""

        # Get reference charges from NAGL
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_charges = [c.m for c in hexane_diol.partial_charges]

        # Add ChargeIncrementModel handler (should provide base charges, not increments)
        increment_handler = ChargeIncrementModelHandler(
            version=0.3,
            partial_charge_method="formal_charge",
        )
        sage_nagl.register_parameter_handler(increment_handler)

        interchange = sage_nagl.create_interchange(topology=hexane_diol.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match NAGL charges, not formal charges
        numpy.testing.assert_allclose(assigned_charges, nagl_charges)


class TestNAGLChargesIntegration:
    """Test NAGLCharges integration with other handlers."""

    def test_nagl_charges_multi_molecule_topology(self, sage_nagl):
        """Test NAGLCharges with multiple molecules in topology."""
        methane = Molecule.from_smiles("C")
        ethane = Molecule.from_smiles("CC")

        topology = Topology.from_molecules([methane, ethane])

        interchange = sage_nagl.create_interchange(topology=topology)
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Each molecule should have approximately zero net charge
        methane_charge_sum = sum(assigned_charges[: methane.n_atoms])
        ethane_charge_sum = sum(assigned_charges[methane.n_atoms :])

        assert abs(methane_charge_sum) < 1e-10 * unit.elementary_charge
        assert abs(ethane_charge_sum) < 1e-10 * unit.elementary_charge

    def test_nagl_charges_with_virtual_sites(self, sage_with_bond_charge):
        """Test NAGLCharges compatibility with virtual sites."""

        # Create a molecule that would have virtual sites
        molecule = Molecule.from_smiles("[Cl]CCO")

        # Add NAGLCharges to the force field
        sage_with_bond_charge.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
                "version": "0.3",
            },
        )

        # Should not raise an error
        interchange = sage_with_bond_charge.create_interchange(
            topology=molecule.to_topology(),
        )

        # Should have charges for real atoms
        assigned_charges = interchange["Electrostatics"]._get_charges()
        assert len(assigned_charges.values()) - 1 == molecule.n_atoms

        # Net charge should be approximately zero
        all_particle_charge_sum = sum(assigned_charges.values())
        assert abs(all_particle_charge_sum) < 1e-10 * unit.elementary_charge
        # Charge without the vsite should be nonzero
        atom_charge_sum = sum([charge for tk, charge in assigned_charges.items() if tk.atom_indices is not None])
        assert abs(atom_charge_sum - (0.123 * unit.elementary_charge)) < 1e-10 * unit.elementary_charge

    def test_nagl_charges_force_field_creation_complete(self, hexane_diol):
        """Test complete interchange creation with NAGLCharges."""

        ff = ForceField("openff-2.1.0.offxml")
        ff.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
                "version": "0.3",
            },
        )

        # Should create complete interchange without errors
        interchange = ff.create_interchange(topology=hexane_diol.to_topology())

        # Should have all expected collections
        expected_collections = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW", "Electrostatics"]
        for collection_name in expected_collections:
            assert collection_name in interchange.collections

        # Electrostatics should have charges
        charges = interchange["Electrostatics"].get_charge_array()
        assert len(charges) == hexane_diol.n_atoms

        # Net charge should be approximately zero
        total_charge = sum(charge.m for charge in charges)
        assert abs(total_charge) < 1e-10

    def test_nagl_charges_identical_molecules_same_charges(self):
        """Test that identical molecules get identical charges from NAGLCharges."""

        # Create topology with two identical molecules
        molecule1 = Molecule.from_smiles("CCO")
        molecule2 = Molecule.from_smiles("CCO")
        topology = Topology.from_molecules([molecule1, molecule2])

        ff = ForceField("openff-2.1.0.offxml")
        ff.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
                "version": "0.3",
            },
        )

        interchange = ff.create_interchange(topology=topology)
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # First molecule charges
        mol1_charges = assigned_charges[: molecule1.n_atoms]
        # Second molecule charges
        mol2_charges = assigned_charges[molecule1.n_atoms :]

        # Should be identical
        numpy.testing.assert_allclose(mol1_charges, mol2_charges)

    def test_nagl_charges_with_charge_from_molecules(self, sage_nagl, hexane_diol):
        """Test that charge_from_molecules takes precedence over NAGLCharges."""
        # Assign preset charges using a different method
        hexane_diol.assign_partial_charges("gasteiger")
        preset_charges = [c.m for c in hexane_diol.partial_charges]

        # Create interchange with charge_from_molecules - should use preset charges
        interchange = sage_nagl.create_interchange(
            topology=hexane_diol.to_topology(),
            charge_from_molecules=[hexane_diol],
        )

        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match preset charges, not NAGL charges
        numpy.testing.assert_allclose(assigned_charges.m, preset_charges)

        # Verify NAGL would give different charges
        hexane_diol_copy = Molecule.from_smiles(hexane_diol.to_smiles())
        hexane_diol_copy.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_charges = [c.m for c in hexane_diol_copy.partial_charges]

        # Preset and NAGL charges should be different
        assert not numpy.allclose(preset_charges, nagl_charges, atol=1e-3)

    def test_nagl_charges_with_mixed_charge_sources(self, sage_nagl):
        """Test NAGLCharges with some molecules having preset charges and others not."""
        # Create molecules
        ethanol = Molecule.from_smiles("CCO")
        methanol = Molecule.from_smiles("CO")

        # Assign preset charges to only one molecule
        ethanol.assign_partial_charges("gasteiger")
        preset_ethanol_charges = [c.m for c in ethanol.partial_charges]

        topology = Topology.from_molecules([ethanol, methanol])

        # Create interchange with preset charges for ethanol only
        interchange = sage_nagl.create_interchange(
            topology=topology,
            charge_from_molecules=[ethanol],
        )

        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # First molecule (ethanol) should match preset charges
        ethanol_charges = assigned_charges[: ethanol.n_atoms]
        numpy.testing.assert_allclose(ethanol_charges.m, preset_ethanol_charges)

        # Second molecule (methanol) should get NAGL charges
        methanol_charges = assigned_charges[ethanol.n_atoms :]

        # Get reference NAGL charges for methanol
        methanol_copy = Molecule.from_smiles("CO")
        methanol_copy.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_methanol_charges = [c.m for c in methanol_copy.partial_charges]

        numpy.testing.assert_allclose(methanol_charges.m, nagl_methanol_charges)

    @pytest.mark.slow
    def test_nagl_charges_large_molecule_performance(self, sage_nagl):
        """Test that NAGL charge assignment completes in reasonable time for large molecules."""
        import time

        # Create a very large molecule
        large_molecule = Molecule.from_smiles("C" * 200)  # 200-carbon alkane chain

        start_time = time.time()

        # Should complete without error
        interchange = sage_nagl.create_interchange(topology=large_molecule.to_topology())

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (less than 30 seconds)
        assert execution_time < 30.0, f"NAGL charge assignment took {execution_time:.2f}s, which is too long"

        # Net charge should be approximately zero
        charges = interchange["Electrostatics"].get_charge_array()
        total_charge = sum(charges.m)
        assert abs(total_charge) < 1e-10

    @pytest.mark.slow
    def test_nagl_charges_multiple_large_molecules_performance(self, sage_nagl):
        """Test performance with multiple large molecules in topology."""
        import time

        # Create multiple copies of medium-sized molecules
        base_molecules = [
            Molecule.from_smiles("C" * 20),  # 20-carbon chain
            Molecule.from_smiles("C" * 25),  # 25-carbon chain
            Molecule.from_smiles("C" * 30),  # 30-carbon chain
        ]

        # Create 20 copies of each
        molecules = []
        for _ in range(20):
            for base_mol in base_molecules:
                molecules.append(base_mol)

        topology = Topology.from_molecules(molecules)

        start_time = time.time()

        # Should complete without error
        interchange = sage_nagl.create_interchange(topology=topology)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert execution_time < 30.0, f"Multi-molecule NAGL assignment took {execution_time:.2f}s, which is too long"

        # Each molecule should have approximately zero net charge
        charges = interchange["Electrostatics"].get_charge_array()
        start_idx = 0
        for molecule in molecules:
            mol_charges = charges[start_idx : start_idx + molecule.n_atoms]
            mol_total_charge = sum(mol_charges.m)
            assert abs(mol_total_charge) < 1e-10
            start_idx += molecule.n_atoms

    @pytest.mark.skip(
        reason="Turn on if toolkit ever allows non-standard scale12/13/15",
    )
    def test_nonstandard_scale_factors(
        sage,
        methane,
    ):
        import random

        # SMIRNOFFSpecError: Current OFF toolkit is unable to handle scale12 values other than 0.0.
        # SMIRNOFFSpecError: Current OFF toolkit is unable to handle scale13 values other than 0.0.
        # SMIRNOFFSpecError: Current OFF toolkit is unable to handle scale15 values other than 1.0.

        factors = {index: random.random() for index in range(2, 6)}

        for handler in ("vdW", "Electrostatics"):
            for index, factor in factors.items():
                setattr(sage[handler], f"scale1{index}", factor)

        interchange = sage.create_interchange(methane.to_topology())

        for collection in ("vdW", "Electrostatics"):
            for index, factor in factors.items():
                assert getattr(interchange[collection], f"scale1{index}") == factor


class TestvdWUpDownConversion:
    def test_upconversion(self):
        handler = vdWHandler(version=0.3, method="cutoff")

        if handler.version == Version("0.4"):
            pytest.skip("Don't test upconversion if the toolkit already did it")

        try:
            _upconvert_vdw_handler(handler)
        except SMIRNOFFVersionError:
            pytest.skip("The installed version of the toolkit does not support 0.4")

        assert handler.version == Version("0.4")
        assert handler.periodic_method == "cutoff"
        assert handler.nonperiodic_method == "no-cutoff"

    def test_downconversion(self):
        try:
            handler = vdWHandler(version=0.4)
        except SMIRNOFFVersionError:
            pytest.skip("The installed version of the toolkit does not support 0.4")

        _downconvert_vdw_handler(handler)

        assert handler.version == Version("0.3")
        assert handler.method == "cutoff"

        # Update when https://github.com/openforcefield/openff-toolkit/issues/1680 is resolved
        try:
            assert not hasattr(handler, "nonperiodic_method")
            assert not hasattr(handler, "periodic_method")
        except AssertionError:
            assert not hasattr(handler, "__delete__")
            pytest.skip("ParameterAttribute.__delete__ not implemented")


class TestElectrostatics:
    def test_caching_detects_atom_ordering(self, sage):
        def get_charges_from_interchange(
            molecule: Molecule,
        ) -> dict[int, Quantity]:
            return {
                key.atom_indices[0]: val
                for key, val in sage.create_interchange(molecule.to_topology())["Electrostatics"]
                ._get_charges()
                .items()
            }

        def compare_charges(
            molecule: Molecule,
            interchange_charges: dict[int, Quantity],
        ):
            for index, molecule_charge in enumerate(molecule.partial_charges):
                assert interchange_charges[index] == molecule_charge

        original = Molecule.from_mapped_smiles("[H:1]-[C:2]#[N:3]")
        reordered = Molecule.from_mapped_smiles("[H:3]-[C:2]#[N:1]")

        for molecule in [original, reordered]:
            molecule.assign_partial_charges("am1bcc")

        compare_charges(original, get_charges_from_interchange(original))
        compare_charges(reordered, get_charges_from_interchange(reordered))

    def test_get_charge_array(self, sage):
        ammonia = Molecule.from_smiles("N")
        ammonia.partial_charges = Quantity(numpy.array([-3, 1, 1, 1]) / 3, "elementary_charge")

        # topology is water | ethanol         | ammonia
        #              O H H C C O H H H H H H N H H H
        topology = Topology.from_molecules(
            [
                Molecule.from_mapped_smiles("[H:2][O:1][H:3]"),
                Molecule.from_mapped_smiles("[H:4][C:1]([H:5])([H:6])[C:2]([H:7])([H:8])[O:3][H:9]"),
                Molecule.from_mapped_smiles("[H:2][N:1]([H:3])[H:4]"),
            ],
        )

        charges = sage.create_interchange(topology, charge_from_molecules=[ammonia])[
            "Electrostatics"
        ].get_charge_array()

        assert isinstance(charges, Quantity)
        assert not isinstance(charges, numpy.ndarray)
        assert isinstance(charges.m, numpy.ndarray)

        assert numpy.allclose(
            charges[:3].m,
            [-0.834, 0.417, 0.417],
        )

        # OpenEye and AmberTools give different AM1-BCC charges
        assert numpy.allclose(
            charges[3:12].m,
            [
                -0.0971,
                0.13143,
                -0.60134,
                0.04476,
                0.04476,
                0.04476,
                0.01732,
                0.01732,
                0.39809,
            ],
        ) or numpy.allclose(
            charges[3:12].m,
            [
                -0.13610011,
                0.12639989,
                -0.59980011,
                0.04236689,
                0.04236689,
                0.04236689,
                0.04319989,
                0.04319989,
                0.39599989,
            ],
        )

        assert numpy.allclose(
            charges[-4:].m,
            ammonia.partial_charges.m,
        )


def test_get_charge_array_fails_if_virtual_sites_present(water, tip4p):
    with pytest.raises(
        NotImplementedError,
        match="Not yet implemented with virtual sites",
    ):
        tip4p.create_interchange(water.to_topology())["Electrostatics"].get_charge_array(include_virtual_sites=True)

    with pytest.raises(
        NotImplementedError,
        match="Not yet implemented when virtual sites are present",
    ):
        tip4p.create_interchange(water.to_topology())["Electrostatics"].get_charge_array(include_virtual_sites=False)


def test_nonintegral_molecule_charge_error(sage, water):
    funky_charges = Quantity([0, 0, -5.5], "elementary_charge")

    water.partial_charges = funky_charges

    with pytest.raises(
        NonIntegralMoleculeChargeError,
        match="net charge of -5.5 compared to a total formal charge of 0.0",
    ):
        sage.create_interchange(water.to_topology(), charge_from_molecules=[water])


class TestSMIRNOFFChargeIncrements:
    @pytest.fixture
    def hydrogen_cyanide_charge_increments(self):
        handler = ChargeIncrementModelHandler(
            version=0.4,
            partial_charge_method="formal_charge",
        )
        handler.add_parameter(
            {
                "smirks": "[H:1][C:2]",
                "charge_increment1": -0.111 * unit.elementary_charge,
                "charge_increment2": 0.111 * unit.elementary_charge,
            },
        )
        handler.add_parameter(
            {
                "smirks": "[C:1]#[N:2]",
                "charge_increment1": 0.5 * unit.elementary_charge,
                "charge_increment2": -0.5 * unit.elementary_charge,
            },
        )

        return handler

    def test_no_charge_increments_applied(self, sage, hexane_diol):
        gastiger_charges = [c.m for c in hexane_diol.partial_charges]
        sage.deregister_parameter_handler("ToolkitAM1BCC")

        no_increments = ChargeIncrementModelHandler(
            version=0.3,
            partial_charge_method="gasteiger",
        )
        sage.register_parameter_handler(no_increments)

        assert len(sage["ChargeIncrementModel"].parameters) == 0

        out = Interchange.from_smirnoff(sage, [hexane_diol])
        assert numpy.allclose(
            numpy.asarray([v.m for v in out["Electrostatics"]._get_charges().values()]),
            gastiger_charges,
        )

    def test_overlapping_increments(self, sage, methane):
        """Test that separate charge increments can be properly applied to the same atom."""
        sage.deregister_parameter_handler("ToolkitAM1BCC")
        charge_handler = ChargeIncrementModelHandler(
            version=0.3,
            partial_charge_method="formal_charge",
        )
        charge_handler.add_parameter(
            {
                "smirks": "[C:1][H:2]",
                "charge_increment1": 0.111 * unit.elementary_charge,
                "charge_increment2": -0.111 * unit.elementary_charge,
            },
        )
        sage.register_parameter_handler(charge_handler)
        assert 0.0 == pytest.approx(
            sum(v.m for v in Interchange.from_smirnoff(sage, [methane])["Electrostatics"]._get_charges().values()),
        )

    def test_charge_increment_forwawrd_reverse_molecule(
        self,
        sage,
        hydrogen_cyanide,
        hydrogen_cyanide_reversed,
        hydrogen_cyanide_charge_increments,
    ):
        sage.deregister_parameter_handler("ToolkitAM1BCC")
        sage.register_parameter_handler(hydrogen_cyanide_charge_increments)

        topology = Topology.from_molecules(
            [hydrogen_cyanide, hydrogen_cyanide_reversed],
        )

        out = Interchange.from_smirnoff(sage, topology)

        expected_charges = [-0.111, 0.611, -0.5, -0.5, 0.611, -0.111]

        # TODO: Fix get_charges to return the atoms in order
        found_charges = [0.0] * topology.n_atoms
        for key, val in out["Electrostatics"]._get_charges().items():
            found_charges[key.atom_indices[0]] = val.m

        assert numpy.allclose(expected_charges, found_charges)
