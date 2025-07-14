import numpy
import pytest
from openff.toolkit import Molecule, Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ToolkitAM1BCCHandler,
    vdWHandler,
)
from openff.toolkit.utils.exceptions import SMIRNOFFVersionError
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

    def test_nagl_charge_assignment_matches_reference(self, sage_with_nagl_charges, hexane_diol):
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        # Leave the ToolkitAM1BCC tag in openff-2.1.0 to ensure that the NAGLCharges handler takes precedence

        interchange = sage_with_nagl_charges.create_interchange(topology=hexane_diol.to_topology())

        assigned_charges_unitless = interchange["Electrostatics"].get_charge_array().m

        expected_charges = hexane_diol.partial_charges
        assert expected_charges is not None
        assert expected_charges.units == unit.elementary_charge
        assert not all(charge == 0 for charge in expected_charges.magnitude)
        expected_charges_unitless = [v.m for v in expected_charges]
        numpy.testing.assert_allclose(expected_charges_unitless, assigned_charges_unitless)


class TestNAGLChargesErrorHandling:
    """Test NAGLCharges error conditions."""

    def test_nagl_charges_missing_toolkit_error(self, sage_with_nagl_charges, hexane_diol):
        """Test MissingPackageError when NAGL toolkit is not available."""
        from openff.toolkit import RDKitToolkitWrapper
        from openff.toolkit.utils.exceptions import MissingPackageError
        from openff.toolkit.utils.toolkit_registry import ToolkitRegistry, toolkit_registry_manager

        # Mock the toolkit registry to not have NAGL
        # RDKit is needed for SMARTS matching.
        with toolkit_registry_manager(ToolkitRegistry(toolkit_precedence=[RDKitToolkitWrapper])):

            with pytest.raises(MissingPackageError, match="NAGL software isn't present"):
                sage_with_nagl_charges.create_interchange(topology=hexane_diol.to_topology())

            # No error should be raised if using charge_from_molecules
            sage_with_nagl_charges.create_interchange(topology=hexane_diol.to_topology(),
                                                      charge_from_molecules=[hexane_diol])


    def test_nagl_charges_invalid_model_file(self, sage, hexane_diol):
        """Test error handling for invalid model file paths."""
        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "nonexistent_model.pt",
                "version": "0.3",
            },
        )
        with pytest.raises(ValueError, match="No registered toolkits can provide the capability"):
            sage.create_interchange(topology=hexane_diol.to_topology())

    def test_nagl_charges_empty_model_file(self, sage, hexane_diol):
        """Test error handling for empty model file parameter."""
        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "",
                "version": "0.3",
            },
        )
        with pytest.raises(ValueError, match="No registered toolkits can provide the capability"):
            sage.create_interchange(topology=hexane_diol.to_topology())

    def test_nagl_charges_none_model_file(self, sage, hexane_diol):
        """Test error handling for None model file parameter."""
        sage.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": None,
                "version": "0.3",
            },
        )
        with pytest.raises(ValueError, match="No registered toolkits can provide the capability"):
            sage.create_interchange(topology=hexane_diol.to_topology())


class TestNAGLChargesPrecedence:
    """Test NAGLCharges precedence in the hierarchy of charge assignment methods."""

    def test_nagl_charges_precedence_over_am1bcc(self, sage_with_nagl_charges, hexane_diol):
        """Test that NAGLCharges takes precedence over ToolkitAM1BCC."""
        # Get reference charges from NAGL
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_charges = [c.m for c in hexane_diol.partial_charges]

        # Get reference charges from AM1BCC
        hexane_diol.assign_partial_charges("am1bcc")
        am1bcc_charges = [c.m for c in hexane_diol.partial_charges]

        # Ensure they're different
        assert not numpy.allclose(nagl_charges, am1bcc_charges)

        interchange = sage_with_nagl_charges.create_interchange(topology=hexane_diol.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match NAGL charges, not AM1BCC
        numpy.testing.assert_allclose(assigned_charges, nagl_charges)

    def test_library_charges_precedence_over_nagl(self, sage_with_nagl_charges, methane):
        """Test that LibraryCharges takes precedence over NAGLCharges."""

        sage_with_nagl_charges["LibraryCharges"].add_parameter(
            {
                "smirks": "[#6X4:1]-[#1:2]",
                "charge1": -0.2 * unit.elementary_charge,
                "charge2": 0.05 * unit.elementary_charge,
            },
        )

        interchange = sage_with_nagl_charges.create_interchange(topology=methane.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match library charges
        expected_charges = [-0.2, 0.05, 0.05, 0.05, 0.05]
        numpy.testing.assert_allclose(assigned_charges, expected_charges)

    def test_nagl_charges_precedence_over_charge_increments(self, sage_with_nagl_charges, hexane_diol):
        """Test that NAGLCharges takes precedence over ChargeIncrementModel as base charges."""
        from openff.toolkit.typing.engines.smirnoff import ChargeIncrementModelHandler

        # Get reference charges from NAGL
        hexane_diol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        nagl_charges = [c.m for c in hexane_diol.partial_charges]

        # Add ChargeIncrementModel handler (should provide base charges, not increments)
        increment_handler = ChargeIncrementModelHandler(
            version=0.3,
            partial_charge_method="formal_charge",
        )
        sage_with_nagl_charges.register_parameter_handler(increment_handler)

        # Remove AM1BCC handler to ensure we're testing NAGL vs ChargeIncrement precedence
        sage_with_nagl_charges.deregister_parameter_handler("ToolkitAM1BCC")

        interchange = sage_with_nagl_charges.create_interchange(topology=hexane_diol.to_topology())
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should match NAGL charges, not formal charges
        numpy.testing.assert_allclose(assigned_charges, nagl_charges)


class TestNAGLChargesIntegration:
    """Test NAGLCharges integration with other handlers."""

    def test_nagl_charges_multi_molecule_topology(self, sage_with_nagl_charges):
        """Test NAGLCharges with multiple molecules in topology."""
        methane = Molecule.from_smiles("C")
        ethane = Molecule.from_smiles("CC")

        topology = Topology.from_molecules([methane, ethane])

        interchange = sage_with_nagl_charges.create_interchange(topology=topology)
        assigned_charges = interchange["Electrostatics"].get_charge_array()

        # Should have charges for all atoms
        assert len(assigned_charges) == topology.n_atoms

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
        from openff.toolkit.typing.engines.smirnoff import ForceField

        from openff.interchange import Interchange

        ff = ForceField("openff-2.1.0.offxml")
        ff.get_parameter_handler(
            "NAGLCharges",
            {
                "model_file": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
                "version": "0.3",
            },
        )

        # Should create complete interchange without errors
        interchange = Interchange.from_smirnoff(force_field=ff, topology=hexane_diol.to_topology())

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
        from openff.toolkit.typing.engines.smirnoff import ForceField

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
