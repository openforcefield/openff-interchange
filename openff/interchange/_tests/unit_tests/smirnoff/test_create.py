import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import get_test_file_path
from openff.interchange.constants import kcal_mol, kcal_mol_a2
from openff.interchange.exceptions import (
    PresetChargesError,
    UnassignedAngleError,
    UnassignedBondError,
    UnassignedTorsionError,
)
from openff.interchange.models import BondKey
from openff.interchange.smirnoff import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
    SMIRNOFFvdWCollection,
)
from openff.interchange.smirnoff._nonbonded import (
    _upconvert_vdw_handler,
    library_charge_from_molecule,
)
from openff.interchange.warnings import PresetChargesAndVirtualSitesWarning


def _get_interpolated_bond_k(bond_handler) -> float:
    for key in bond_handler.key_map:
        if key.bond_order is not None:
            topology_key = key
            break
    potential_key = bond_handler.key_map[topology_key]
    return bond_handler.potentials[potential_key].parameters["k"].m


class TestCreate:
    def test_modified_nonbonded_cutoffs(self, sage, ethanol):
        topology = Topology.from_molecules(ethanol)
        modified_sage = ForceField(sage.to_string())

        modified_sage["vdW"].cutoff = 0.777 * unit.angstrom
        modified_sage["Electrostatics"].cutoff = 0.777 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=modified_sage, topology=topology)

        assert out["vdW"].cutoff == 0.777 * unit.angstrom
        assert out["Electrostatics"].cutoff == 0.777 * unit.angstrom

    def test_sage_tip3p_charges(self, water, sage):
        """Ensure tip3p charges packaged with sage are applied over AM1-BCC charges.
        https://github.com/openforcefield/openff-toolkit/issues/1199"""
        out = Interchange.from_smirnoff(force_field=sage, topology=[water])
        found_charges = [v.m for v in out["Electrostatics"]._get_charges().values()]

        assert numpy.allclose(found_charges, [-0.834, 0.417, 0.417])

    def test_infer_positions(self, sage, ethanol):
        assert Interchange.from_smirnoff(sage, [ethanol]).positions is None

        ethanol.generate_conformers(n_conformers=1)

        assert Interchange.from_smirnoff(sage, [ethanol]).positions.shape == (
            ethanol.n_atoms,
            3,
        )

    def test_cosmetic_attributes(self):
        from openff.toolkit._tests.test_forcefield import xml_ff_w_cosmetic_elements

        force_field = ForceField(
            "openff-2.1.0.offxml",
            xml_ff_w_cosmetic_elements.replace(
                'Bonds version="0.3"',
                'Bonds version="0.4"',
            ),
            allow_cosmetic_attributes=True,
        )

        bonds = SMIRNOFFBondCollection()

        bonds.store_matches(
            parameter_handler=force_field["Bonds"],
            topology=Molecule.from_smiles("CC").to_topology(),
        )

        for key in bonds.potentials:
            if key.id == "[#6X4:1]-[#6X4:2]":
                assert key.cosmetic_attributes == {
                    "parameters": "k, length",
                    "parameterize_eval": "blah=blah2",
                }

    def test_all_cosmetic(self, sage, basic_top):
        for handler in sage.registered_parameter_handlers:
            for parameter in sage[handler].parameters:
                parameter._cosmetic_attribs = ["fOO"]
                parameter._fOO = "bAR"
                parameter.fOO = "bAR"

        out = sage.create_interchange(basic_top)

        for collection in out.collections:
            if collection == "Electrostatics":
                continue

            for potential_key in out[collection].potentials:
                assert potential_key.cosmetic_attributes["fOO"] == "bAR"


class TestUnassignedParameters:
    def test_catch_unassigned_bonds(self, sage, ethanol_top):
        for param in sage["Bonds"].parameters:
            param.smirks = "[#99:1]-[#99:2]"

        sage.deregister_parameter_handler(sage["Constraints"])

        with pytest.raises(
            UnassignedBondError,
            match="BondHandler was not able to find par",
        ):
            Interchange.from_smirnoff(force_field=sage, topology=ethanol_top)

    def test_catch_unassigned_angles(self, sage, ethanol_top):
        for param in sage["Angles"].parameters:
            param.smirks = "[#99:1]-[#99:2]-[#99:3]"

        with pytest.raises(
            UnassignedAngleError,
            match="AngleHandler was not able to find par",
        ):
            Interchange.from_smirnoff(force_field=sage, topology=ethanol_top)

    def test_catch_unassigned_torsions(self, sage, ethanol_top):
        for param in sage["ProperTorsions"].parameters:
            param.smirks = "[#99:1]-[#99:2]-[#99:3]-[#99:4]"

        with pytest.raises(
            UnassignedTorsionError,
            match="- Topology indices [(]5, 0, 1, 6[)]: "
            r"names and elements [(](H\d+)? H[)], [(](C\d+)? C[)], [(](C\d+)? C[)], [(](H\d+)? H[)],",
        ):
            Interchange.from_smirnoff(force_field=sage, topology=ethanol_top)


# TODO: Remove xfail after openff-toolkit 0.10.0
@pytest.mark.xfail
def test_library_charges_from_molecule():
    from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler

    mol = Molecule.from_mapped_smiles("[Cl:1][C:2]#[C:3][F:4]")

    with pytest.raises(ValueError, match="missing partial"):
        library_charge_from_molecule(mol)

    mol.partial_charges = numpy.linspace(-0.3, 0.3, 4) * unit.elementary_charge

    library_charges = library_charge_from_molecule(mol)

    assert isinstance(library_charges, LibraryChargeHandler.LibraryChargeType)
    assert library_charges.smirks == mol.to_smiles(mapped=True)
    assert library_charges.charge == [*mol.partial_charges]


class TestPresetCharges:
    @pytest.mark.slow
    def test_charge_from_molecules_basic(self, sage):
        molecule = Molecule.from_smiles("CCO")
        molecule.assign_partial_charges(partial_charge_method="am1bcc")
        molecule.partial_charges *= -1

        default = Interchange.from_smirnoff(sage, molecule.to_topology())
        uses = Interchange.from_smirnoff(
            sage,
            molecule.to_topology(),
            charge_from_molecules=[molecule],
        )

        found_charges_no_uses = [v.m for v in default["Electrostatics"]._get_charges().values()]
        found_charges_uses = [v.m for v in uses["Electrostatics"]._get_charges().values()]

        assert not numpy.allclose(found_charges_no_uses, found_charges_uses)

        assert numpy.allclose(found_charges_uses, molecule.partial_charges.m)

    def test_charges_on_molecules_in_topology(self, sage, water):
        ethanol = Molecule.from_smiles("CCO")

        ethanol_charges = numpy.linspace(-1, 1, 9) * 0.4
        water_charges = numpy.linspace(-1, 1, 3)

        ethanol.partial_charges = Quantity(ethanol_charges, unit.elementary_charge)
        water.partial_charges = Quantity(water_charges, unit.elementary_charge)

        out = Interchange.from_smirnoff(
            sage,
            [ethanol],
            charge_from_molecules=[ethanol, water],
        )

        for molecule in out.topology.molecules:
            if "C" in molecule.to_smiles():
                assert numpy.allclose(molecule.partial_charges.m, ethanol_charges)
            else:
                assert numpy.allclose(molecule.partial_charges.m, water_charges)

    def test_charges_from_molecule_reordered(
        self,
        sage,
        hydrogen_cyanide,
        hydrogen_cyanide_reversed,
    ):
        """Test the behavior of charge_from_molecules when the atom ordering differs with the topology"""

        # H - C # N
        molecule = hydrogen_cyanide

        #  N  # C  - H
        # -0.3, 0.0, 0.3
        molecule_with_charges = hydrogen_cyanide_reversed
        molecule_with_charges.partial_charges = Quantity(
            [-0.3, 0.0, 0.3],
            unit.elementary_charge,
        )

        out = Interchange.from_smirnoff(
            sage,
            molecule.to_topology(),
            charge_from_molecules=[molecule_with_charges],
        )

        expected_charges = [0.3, 0.0, -0.3]
        found_charges = [v.m for v in out["Electrostatics"]._get_charges().values()]

        assert numpy.allclose(expected_charges, found_charges)

    def test_error_when_any_missing_partial_charges(self, sage):
        topology = Topology.from_molecules(
            [
                Molecule.from_smiles("C"),
                Molecule.from_smiles("CCO"),
            ],
        )

        with pytest.raises(
            PresetChargesError,
            match="All.*must have partial charges",
        ):
            sage.create_interchange(
                topology,
                charge_from_molecules=[Molecule.from_smiles("C")],
            )

    def test_error_multiple_isomorphic_molecules(self, sage, ethanol, reversed_ethanol):
        # these ethanol fixtures *do* have charges, if they didn't have charges it would be
        # ambiguous which error should be raised
        topology = Topology.from_molecules([ethanol, reversed_ethanol, ethanol])

        with pytest.raises(
            PresetChargesError,
            match="All molecules.*isomorphic.*unique",
        ):
            sage.create_interchange(
                topology,
                charge_from_molecules=[ethanol, reversed_ethanol],
            )

    def test_virtual_site_charge_increments_applied_after_preset_charges_water(
        self,
        tip4p,
        water,
    ):
        """
        Test that charge increments to/from virtual sites are still applied after preset charges are set.

        This is funky user input, since preset charges override library charges, which are important for water.
        """
        water.partial_charges = Quantity(
            [-2.0, 1.0, 1.0],
            "elementary_charge",
        )

        with pytest.warns(PresetChargesAndVirtualSitesWarning):
            # This dict is probably ordered O H H VS
            charges = tip4p.create_interchange(
                water.to_topology(),
                charge_from_molecules=[water],
            )["Electrostatics"].charges

        # tip4p_fb.offxml is meant to result in charges of -1.0517362213526 (VS) 0 (O) and 0.5258681106763 (H)
        # the force field has 0.0 for all library charges (skipping AM1-BCC), so preset charges screw this up.
        # the resulting charges, if using preset charges of [-2, 1, 1], should be the result of adding together
        # [-2, 1, 1, 0] (preset charges) and
        # [0, 1.5258681106763001, 1.5258681106763001, -1.0517362213526] (charge increments)

        numpy.testing.assert_allclose(
            [charge.m for charge in charges.values()],
            [-2.0, 1.5258681106763001, 1.5258681106763001, -1.0517362213526],
        )

    def test_virtual_site_charge_increments_applied_after_preset_charges_ligand(
        self,
        sage_with_sigma_hole,
    ):
        ligand = Molecule.from_mapped_smiles("[C:1]([Cl:2])([Cl:3])([Cl:4])[Cl:5]")

        ligand.partial_charges = Quantity(
            [1.2, -0.3, -0.3, -0.3, -0.3],
            "elementary_charge",
        )

        with pytest.warns(PresetChargesAndVirtualSitesWarning):
            # This dict is probably ordered C Cl Cl Cl Cl VS VS VS VS
            charges = sage_with_sigma_hole.create_interchange(
                ligand.to_topology(),
                charge_from_molecules=[ligand],
            )["Electrostatics"].charges

        # this fake sigma hole parameter shifts 0.1 e from the carbon and 0.2 e from the chlorine, so the
        # resulting charges should be like
        # [1.2, -0.3, -0.3, -0.3, -0.3] +
        # [0.4, 0.2, 0.2, 0.2, 0.2, -0.3, -0.3, -0.3, -0.3]

        numpy.testing.assert_allclose(
            [charge.m for charge in charges.values()],
            [1.6, -0.1, -0.1, -0.1, -0.1, -0.3, -0.3, -0.3, -0.3],
        )


@skip_if_missing("openmm")
class TestPartialBondOrdersFromMolecules:
    @pytest.mark.parametrize(
        (
            "reversed",
            "central_atoms",
        ),
        [
            (False, (1, 2)),
            (True, (7, 6)),
        ],
    )
    def test_interpolated_partial_bond_orders_from_molecules(
        self,
        reversed,
        central_atoms,
        ethanol,
        reversed_ethanol,
    ):
        """Test the fractional bond orders are used to interpolate k and length values as we expect,
        including that the fractional bond order is defined by the value on the input molecule via
        `partial_bond_order_from_molecules`, not whatever is produced by a default call to
        `Molecule.assign_fractional_bond_orders`.

        This test is adapted from test_fractional_bondorder_from_molecule in the toolkit.

        Parameter   | param values at bond orders 1, 2  | used bond order   | expected value
        bond k        101, 123 kcal/mol/A**2              1.55                113.1 kcal/mol/A**2
        bond length   1.4, 1.3 A                          1.55                1.345 A
        torsion k     1, 1.8 kcal/mol                     1.55                1.44 kcal/mol
        """
        from openff.toolkit._tests.test_forcefield import xml_ff_bo

        molecule = reversed_ethanol if reversed else ethanol

        molecule.get_bond_between(*central_atoms).fractional_bond_order = 1.55

        sorted_indices = tuple(sorted(central_atoms))

        out = Interchange.from_smirnoff(
            force_field=ForceField(
                "openff-2.0.0.offxml",
                xml_ff_bo,
            ),
            topology=molecule.to_topology(),
            partial_bond_orders_from_molecules=[molecule],
        )

        bond_key = BondKey(atom_indices=sorted_indices, bond_order=1.55)
        bond_potential = out["Bonds"].key_map[bond_key]
        found_bond_k = out["Bonds"].potentials[bond_potential].parameters["k"]
        found_bond_length = out["Bonds"].potentials[bond_potential].parameters["length"]

        assert found_bond_k.m_as(kcal_mol_a2) == pytest.approx(113.1)
        assert found_bond_length.m_as(unit.angstrom) == pytest.approx(1.345)

        # TODO: There should be a better way of plucking this torsion's TopologyKey
        for topology_key in out["ProperTorsions"].key_map.keys():
            if (tuple(sorted(topology_key.atom_indices))[1:3] == sorted_indices) and topology_key.bond_order == 1.55:
                torsion_key = topology_key
                break

        torsion_potential = out["ProperTorsions"].key_map[torsion_key]
        found_torsion_k = out["ProperTorsions"].potentials[torsion_potential].parameters["k"]

        assert found_torsion_k.m_as(kcal_mol) == pytest.approx(1.44)

    @pytest.mark.slow
    def test_partial_bond_order_from_molecules_empty(self, ethanol):
        from openff.toolkit._tests.test_forcefield import xml_ff_bo

        forcefield = ForceField(
            "openff-2.0.0.offxml",
            xml_ff_bo,
        )

        default = Interchange.from_smirnoff(forcefield, ethanol.to_topology())
        empty = Interchange.from_smirnoff(
            forcefield,
            ethanol.to_topology(),
            partial_bond_orders_from_molecules=list(),
        )

        assert _get_interpolated_bond_k(default["Bonds"]) == pytest.approx(
            _get_interpolated_bond_k(empty["Bonds"]),
        )

    def test_partial_bond_order_from_molecules_no_matches(self, ethanol):
        from openff.toolkit._tests.test_forcefield import xml_ff_bo

        forcefield = ForceField(
            "openff-2.0.0.offxml",
            xml_ff_bo,
        )

        decoy = Molecule.from_smiles("C#N")
        decoy.assign_fractional_bond_orders(bond_order_model="am1-wiberg")

        default = Interchange.from_smirnoff(forcefield, ethanol.to_topology())
        uses = Interchange.from_smirnoff(
            forcefield,
            ethanol.to_topology(),
            partial_bond_orders_from_molecules=[decoy],
        )

        assert _get_interpolated_bond_k(default["Bonds"]) == pytest.approx(
            _get_interpolated_bond_k(uses["Bonds"]),
        )


class TestCreateWithPlugins:
    def test_setup_plugins(self):
        from nonbonded_plugins.nonbonded import (
            BuckinghamHandler,
            SMIRNOFFBuckinghamCollection,
        )

        from openff.interchange.smirnoff._create import _PLUGIN_CLASS_MAPPING

        assert _PLUGIN_CLASS_MAPPING[BuckinghamHandler] == SMIRNOFFBuckinghamCollection

    def test_create_buckingham(self, water):
        force_field = ForceField(
            get_test_file_path("buckingham.offxml"),
            load_plugins=True,
        )

        out = Interchange.from_smirnoff(
            force_field,
            water.to_topology(),
        )

        assert "Buckingham" in out.collections
        assert len(out["Buckingham"].potentials) == 2


@skip_if_missing("jax")
class TestMatrixRepresentations:
    @pytest.mark.parametrize(
        ("handler_name", "n_ff_terms", "n_sys_terms"),
        [("vdW", 10, 72), ("Bonds", 8, 64), ("Angles", 6, 104)],
    )
    def test_to_force_field_to_system_parameters(
        self,
        sage,
        ethanol_top,
        handler_name,
        n_ff_terms,
        n_sys_terms,
    ):
        import jax

        if handler_name == "Bonds":
            handler = SMIRNOFFBondCollection.create(
                parameter_handler=sage["Bonds"],
                topology=ethanol_top,
            )
        elif handler_name == "Angles":
            handler = SMIRNOFFAngleCollection.create(
                parameter_handler=sage[handler_name],
                topology=ethanol_top,
            )
        elif handler_name == "vdW":
            # Ensure this is 0.3 until everything assumes 0.4
            _upconvert_vdw_handler(sage[handler_name])

            handler = SMIRNOFFvdWCollection.create(
                parameter_handler=sage[handler_name],
                topology=ethanol_top,
            )
        else:
            raise NotImplementedError()

        p = handler.get_force_field_parameters(use_jax=True)

        assert isinstance(p, jax.Array)
        assert numpy.prod(p.shape) == n_ff_terms

        q = handler.get_system_parameters(use_jax=True)

        assert isinstance(q, jax.Array)
        assert numpy.prod(q.shape) == n_sys_terms

        assert jax.numpy.allclose(q, handler.parametrize(p))

        param_matrix = handler.get_param_matrix()

        ref_file = get_test_file_path(f"ethanol_param_{handler_name.lower()}.npy")
        ref = jax.numpy.load(ref_file)

        assert jax.numpy.allclose(ref, param_matrix)

        # TODO: Update with other handlers that can safely be assumed to follow 1:1 slot:smirks mapping
        if handler_name in ["vdW", "Bonds", "Angles"]:
            assert numpy.allclose(
                numpy.sum(param_matrix, axis=1),
                numpy.ones(param_matrix.shape[0]),
            )

    def test_set_force_field_parameters(self, sage, ethanol):
        import jax

        bond_handler = SMIRNOFFBondCollection.create(
            parameter_handler=sage["Bonds"],
            topology=ethanol.to_topology(),
        )

        original = bond_handler.get_force_field_parameters(use_jax=True)
        modified = original * jax.numpy.array([1.1, 0.5])

        bond_handler.set_force_field_parameters(modified)

        assert (bond_handler.get_force_field_parameters(use_jax=True) == modified).all()
