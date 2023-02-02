import itertools
from typing import List, Tuple

import numpy
import openmm
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterHandler,
    VirtualSiteHandler,
    vdWHandler,
)
from openff.units import unit
from openff.units.openmm import to_openmm
from openff.utilities.testing import skip_if_missing
from openmm import unit as openmm_unit

from openff.interchange import Interchange
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    UnassignedAngleError,
    UnassignedBondError,
    UnassignedTorsionError,
)
from openff.interchange.models import BondKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFvdWCollection,
    library_charge_from_molecule,
)
from openff.interchange.smirnoff._valence import (
    SMIRNOFFAngleCollection,
    SMIRNOFFBondCollection,
)
from openff.interchange.tests import _BaseTest, get_test_file_path

kcal_mol = unit.Unit("kilocalorie / mol")
kcal_mol_a2 = unit.Unit("kilocalorie / (angstrom ** 2 * mole)")
kcal_mol_rad2 = unit.Unit("kilocalorie / (mole * radian ** 2)")


def _get_interpolated_bond_k(bond_handler) -> float:
    for key in bond_handler.slot_map:
        if key.bond_order is not None:
            topology_key = key
            break
    potential_key = bond_handler.slot_map[topology_key]
    return bond_handler.potentials[potential_key].parameters["k"].m


class TestSMIRNOFFCollection(_BaseTest):
    def test_allowed_parameter_handler_types(self):
        class DummyParameterHandler(ParameterHandler):
            pass

        class DummySMIRNOFFCollection(SMIRNOFFCollection):
            type = "Bonds"
            expression = "1+1"

            @classmethod
            def allowed_parameter_handlers(cls):
                return [DummyParameterHandler]

            @classmethod
            def supported_parameters(cls):
                return list()

        dummy_handler = DummySMIRNOFFCollection()
        angle_Handler = AngleHandler(version=0.3)

        assert DummyParameterHandler in dummy_handler.allowed_parameter_handlers()
        assert AngleHandler not in dummy_handler.allowed_parameter_handlers()
        assert (
            DummyParameterHandler
            not in SMIRNOFFAngleCollection.allowed_parameter_handlers()
        )

        dummy_handler = DummyParameterHandler(version=0.3)

        with pytest.raises(InvalidParameterHandlerError):
            SMIRNOFFAngleCollection.create(
                parameter_handler=dummy_handler,
                topology=Topology(),
            )

        with pytest.raises(InvalidParameterHandlerError):
            DummySMIRNOFFCollection.create(
                parameter_handler=angle_Handler,
                topology=Topology(),
            )


class TestInterchangeFromSMIRNOFF(_BaseTest):
    """General tests for Interchange.from_smirnoff. Some are ported from the toolkit."""

    def test_modified_nonbonded_cutoffs(self, sage):
        from openff.toolkit.tests.create_molecules import create_ethanol

        topology = Topology.from_molecules(create_ethanol())
        modified_sage = ForceField(sage.to_string())

        modified_sage["vdW"].cutoff = 0.777 * unit.angstrom
        modified_sage["Electrostatics"].cutoff = 0.777 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=modified_sage, topology=topology)

        assert out["vdW"].cutoff == 0.777 * unit.angstrom
        assert out["Electrostatics"].cutoff == 0.777 * unit.angstrom

    def test_sage_tip3p_charges(self, tip3p_xml):
        """Ensure tip3p charges packaged with sage are applied over AM1-BCC charges.
        https://github.com/openforcefield/openff-toolkit/issues/1199"""
        sage = ForceField("openff-2.0.0.offxml")

        topology = Molecule.from_smiles("O").to_topology()
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        out = Interchange.from_smirnoff(force_field=sage, topology=topology)
        found_charges = [v.m for v in out["Electrostatics"].charges.values()]

        assert numpy.allclose(found_charges, [-0.834, 0.417, 0.417])

    def test_infer_positions(self, sage):
        from openff.toolkit.tests.create_molecules import create_ethanol

        molecule = create_ethanol()

        assert Interchange.from_smirnoff(sage, [molecule]).positions is None

        molecule.generate_conformers(n_conformers=1)

        assert Interchange.from_smirnoff(sage, [molecule]).positions.shape == (
            molecule.n_atoms,
            3,
        )


@pytest.mark.slow()
class TestUnassignedParameters(_BaseTest):
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
@pytest.mark.xfail()
def test_library_charges_from_molecule():
    mol = Molecule.from_mapped_smiles("[Cl:1][C:2]#[C:3][F:4]")

    with pytest.raises(ValueError, match="missing partial"):
        library_charge_from_molecule(mol)

    mol.partial_charges = numpy.linspace(-0.3, 0.3, 4) * unit.elementary_charge

    library_charges = library_charge_from_molecule(mol)

    assert isinstance(library_charges, LibraryChargeHandler.LibraryChargeType)
    assert library_charges.smirks == mol.to_smiles(mapped=True)
    assert library_charges.charge == [*mol.partial_charges]


class TestChargeFromMolecules(_BaseTest):
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

        found_charges_no_uses = [
            v.m for v in default["Electrostatics"].charges.values()
        ]
        found_charges_uses = [v.m for v in uses["Electrostatics"].charges.values()]

        assert not numpy.allclose(found_charges_no_uses, found_charges_uses)

        assert numpy.allclose(found_charges_uses, molecule.partial_charges.m)

    def test_charges_on_molecules_in_topology(self, sage):
        ethanol = Molecule.from_smiles("CCO")
        water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
        ethanol_charges = numpy.linspace(-1, 1, 9) * 0.4
        water_charges = numpy.linspace(-1, 1, 3)

        ethanol.partial_charges = unit.Quantity(ethanol_charges, unit.elementary_charge)
        water.partial_charges = unit.Quantity(water_charges, unit.elementary_charge)

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
        molecule_with_charges.partial_charges = unit.Quantity(
            [-0.3, 0.0, 0.3],
            unit.elementary_charge,
        )

        out = Interchange.from_smirnoff(
            sage,
            molecule.to_topology(),
            charge_from_molecules=[molecule_with_charges],
        )

        expected_charges = [0.3, 0.0, -0.3]
        found_charges = [v.m for v in out["Electrostatics"].charges.values()]

        assert numpy.allclose(expected_charges, found_charges)


class TestPartialBondOrdersFromMolecules(_BaseTest):
    from openff.toolkit.tests.create_molecules import (
        create_ethanol,
        create_reversed_ethanol,
    )

    @pytest.mark.parametrize(
        (
            "get_molecule",
            "central_atoms",
        ),
        [
            (create_ethanol, (1, 2)),
            (create_reversed_ethanol, (7, 6)),
        ],
    )
    def test_interpolated_partial_bond_orders_from_molecules(
        self,
        get_molecule,
        central_atoms,
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
        mol = get_molecule()
        mol.get_bond_between(*central_atoms).fractional_bond_order = 1.55

        sorted_indices = tuple(sorted(central_atoms))

        from openff.toolkit.tests.test_forcefield import xml_ff_bo

        forcefield = ForceField("test_forcefields/test_forcefield.offxml", xml_ff_bo)
        topology = Topology.from_molecules(mol)

        out = Interchange.from_smirnoff(
            force_field=forcefield,
            topology=topology,
            partial_bond_orders_from_molecules=[mol],
        )

        bond_key = BondKey(atom_indices=sorted_indices, bond_order=1.55)
        bond_potential = out["Bonds"].slot_map[bond_key]
        found_bond_k = out["Bonds"].potentials[bond_potential].parameters["k"]
        found_bond_length = out["Bonds"].potentials[bond_potential].parameters["length"]

        assert found_bond_k.m_as(kcal_mol_a2) == pytest.approx(113.1)
        assert found_bond_length.m_as(unit.angstrom) == pytest.approx(1.345)

        # TODO: There should be a better way of plucking this torsion's TopologyKey
        for topology_key in out["ProperTorsions"].slot_map.keys():
            if (
                tuple(sorted(topology_key.atom_indices))[1:3] == sorted_indices
            ) and topology_key.bond_order == 1.55:
                torsion_key = topology_key
                break

        torsion_potential = out["ProperTorsions"].slot_map[torsion_key]
        found_torsion_k = (
            out["ProperTorsions"].potentials[torsion_potential].parameters["k"]
        )

        assert found_torsion_k.m_as(kcal_mol) == pytest.approx(1.44)

    def test_partial_bond_order_from_molecules_empty(self):
        from openff.toolkit.tests.test_forcefield import xml_ff_bo

        forcefield = ForceField("test_forcefields/test_forcefield.offxml", xml_ff_bo)

        molecule = Molecule.from_smiles("CCO")

        default = Interchange.from_smirnoff(forcefield, molecule.to_topology())
        empty = Interchange.from_smirnoff(
            forcefield,
            molecule.to_topology(),
            partial_bond_orders_from_molecules=list(),
        )

        assert _get_interpolated_bond_k(default["Bonds"]) == pytest.approx(
            _get_interpolated_bond_k(empty["Bonds"]),
        )

    def test_partial_bond_order_from_molecules_no_matches(self):
        from openff.toolkit.tests.test_forcefield import xml_ff_bo

        forcefield = ForceField("test_forcefields/test_forcefield.offxml", xml_ff_bo)

        molecule = Molecule.from_smiles("CCO")
        decoy = Molecule.from_smiles("C#N")
        decoy.assign_fractional_bond_orders(bond_order_model="am1-wiberg")

        default = Interchange.from_smirnoff(forcefield, molecule.to_topology())
        uses = Interchange.from_smirnoff(
            forcefield,
            molecule.to_topology(),
            partial_bond_orders_from_molecules=[decoy],
        )

        assert _get_interpolated_bond_k(default["Bonds"]) == pytest.approx(
            _get_interpolated_bond_k(uses["Bonds"]),
        )


@skip_if_missing("jax")
class TestMatrixRepresentations(_BaseTest):
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


class TestSMIRNOFFVirtualSites(_BaseTest):
    from openff.toolkit.tests.mocking import VirtualSiteMocking
    from openmm import unit as openmm_unit

    @classmethod
    def build_force_field(cls, v_site_handler: VirtualSiteHandler) -> ForceField:
        force_field = ForceField()

        force_field.get_parameter_handler("Bonds").add_parameter(
            {
                "smirks": "[*:1]~[*:2]",
                "k": 0.0 * unit.kilojoule_per_mole / unit.angstrom**2,
                "length": 0.0 * unit.angstrom,
            },
        )
        force_field.get_parameter_handler("Angles").add_parameter(
            {
                "smirks": "[*:1]~[*:2]~[*:3]",
                "k": 0.0 * unit.kilojoule_per_mole / unit.degree**2,
                "angle": 60.0 * unit.degree,
            },
        )
        force_field.get_parameter_handler("ProperTorsions").add_parameter(
            {
                "smirks": "[*:1]~[*:2]~[*:3]~[*:4]",
                "k": [0.0] * unit.kilojoule_per_mole,
                "phase": [0.0] * unit.degree,
                "periodicity": [2],
                "idivf": [1.0],
            },
        )
        force_field.get_parameter_handler("vdW").add_parameter(
            {
                "smirks": "[*:1]",
                "epsilon": 0.0 * unit.kilojoule_per_mole,
                "sigma": 1.0 * unit.angstrom,
            },
        )
        force_field.get_parameter_handler("LibraryCharges").add_parameter(
            {"smirks": "[*:1]", "charge": [0.0] * unit.elementary_charge},
        )
        force_field.get_parameter_handler("Electrostatics")

        force_field.register_parameter_handler(v_site_handler)

        return force_field

    @classmethod
    def generate_v_site_coordinates(
        cls,
        molecule: Molecule,
        input_conformer: unit.Quantity,
        parameter: VirtualSiteHandler.VirtualSiteType,
    ) -> unit.Quantity:
        # Compute the coordinates of the virtual site. Unfortunately OpenMM does not
        # seem to offer a more compact way to do this currently.
        handler = VirtualSiteHandler(version="0.3")
        handler.add_parameter(parameter=parameter)

        force_field = cls.build_force_field(handler)

        system = Interchange.from_smirnoff(
            force_field=force_field,
            topology=molecule.to_topology(),
        ).to_openmm(combine_nonbonded_forces=True)

        n_v_sites = sum(
            1 if system.isVirtualSite(i) else 0 for i in range(system.getNumParticles())
        )

        input_conformer = unit.Quantity(
            numpy.vstack(
                [
                    input_conformer.m_as(unit.angstrom),
                    numpy.zeros((n_v_sites, 3)),
                ],
            ),
            unit.angstrom,
        )

        context = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0 * openmm_unit.femtosecond),
            openmm.Platform.getPlatformByName("Reference"),
        )

        context.setPositions(to_openmm(input_conformer))
        context.computeVirtualSites()

        output_conformer = context.getState(getPositions=True).getPositions(
            asNumpy=True,
        )

        return output_conformer[molecule.n_atoms :, :]

    @pytest.mark.parametrize(
        (
            "parameter",
            "smiles",
            "input_conformer",
            "atoms_to_shuffle",
            "expected_coordinates",
        ),
        [
            (
                VirtualSiteMocking.bond_charge_parameter("[Cl:1]-[C:2]"),
                "[Cl:1][C:2]([H:3])([H:4])[H:5]",
                VirtualSiteMocking.sp3_conformer(),
                (0, 1),
                unit.Quantity(numpy.array([[0.0, 3.0, 0.0]]), unit.angstrom),
            ),
            (
                VirtualSiteMocking.bond_charge_parameter("[C:1]#[C:2]"),
                "[H:1][C:2]#[C:3][H:4]",
                VirtualSiteMocking.sp1_conformer(),
                (2, 3),
                unit.Quantity(
                    numpy.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                    unit.angstrom,
                ),
            ),
            (
                VirtualSiteMocking.monovalent_parameter("[O:1]=[C:2]-[H:3]"),
                "[O:1]=[C:2]([H:3])[H:4]",
                VirtualSiteMocking.sp2_conformer(),
                (0, 1, 2, 3),
                (
                    VirtualSiteMocking.sp2_conformer()[0]
                    + unit.Quantity(  # noqa
                        numpy.array(
                            [[1.0, numpy.sqrt(2), 1.0], [1.0, -numpy.sqrt(2), -1.0]],
                        ),
                        unit.angstrom,
                    )
                ),
            ),
            (
                VirtualSiteMocking.divalent_parameter(
                    "[H:2][O:1][H:3]",
                    match="once",
                    angle=0.0 * unit.degree,
                ),
                "[H:2][O:1][H:3]",
                VirtualSiteMocking.sp2_conformer()[1:, :],
                (0, 1, 2),
                numpy.array([[2.0, 0.0, 0.0]]) * unit.angstrom,
            ),
            (
                VirtualSiteMocking.divalent_parameter(
                    "[H:2][O:1][H:3]",
                    match="all_permutations",
                    angle=45.0 * unit.degree,
                ),
                "[H:2][O:1][H:3]",
                VirtualSiteMocking.sp2_conformer()[1:, :],
                (0, 1, 2),
                unit.Quantity(
                    numpy.array(
                        [
                            [numpy.sqrt(2), numpy.sqrt(2), 0.0],
                            [numpy.sqrt(2), -numpy.sqrt(2), 0.0],
                        ],
                    ),
                    unit.angstrom,
                ),
            ),
            (
                VirtualSiteMocking.trivalent_parameter("[N:1]([H:2])([H:3])[H:4]"),
                "[N:1]([H:2])([H:3])[H:4]",
                VirtualSiteMocking.sp3_conformer()[1:, :],
                (0, 1, 2, 3),
                numpy.array([[0.0, 2.0, 0.0]]) * unit.angstrom,
            ),
        ],
    )
    def test_v_site_geometry(
        self,
        parameter: VirtualSiteHandler.VirtualSiteType,
        smiles: str,
        input_conformer: unit.Quantity,
        atoms_to_shuffle: Tuple[int, ...],
        expected_coordinates: unit.Quantity,
    ):
        """An integration test that virtual sites are placed correctly relative to the
        parent atoms"""

        for atom_permutation in itertools.permutations(atoms_to_shuffle):
            molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            molecule._conformers = [input_conformer]

            # We shuffle the ordering of the atoms involved in the v-site orientation
            # to ensure that the orientation remains invariant as expected.
            shuffled_atom_order = {i: i for i in range(molecule.n_atoms)}
            shuffled_atom_order.update(
                {
                    old_index: new_index
                    for old_index, new_index in zip(atoms_to_shuffle, atom_permutation)
                },
            )

            molecule = molecule.remap(shuffled_atom_order)

            output_coordinates = self.generate_v_site_coordinates(
                molecule,
                molecule.conformers[0],
                parameter,
            )

            assert output_coordinates.shape == expected_coordinates.shape

            def sort_coordinates(x: numpy.ndarray) -> numpy.ndarray:
                # Sort the rows by first, then second, then third columns as row
                # as the order of v-sites is not deterministic.
                return x[numpy.lexsort((x[:, 2], x[:, 1], x[:, 0])), :]

            found = sort_coordinates(
                output_coordinates.value_in_unit(openmm_unit.angstrom),
            )
            expected = sort_coordinates(expected_coordinates.m_as(unit.angstrom))

            assert numpy.allclose(found, expected), expected - found

    _E = unit.elementary_charge
    _A = unit.angstrom
    _KJ = unit.kilojoule_per_mole

    @pytest.mark.parametrize(
        ("topology", "parameters", "expected_parameters", "expected_n_v_sites"),
        [
            (
                Topology.from_molecules(
                    [
                        VirtualSiteMocking.chloromethane(reverse=False),
                        VirtualSiteMocking.chloromethane(reverse=True),
                    ],
                ),
                [VirtualSiteMocking.bond_charge_parameter("[Cl:1][C:2]")],
                [
                    # charge, sigma, epsilon
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.3 * _E, 4.0 * _A, 3.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.3 * _E, 4.0 * _A, 3.0 * _KJ),
                ],
                2,
            ),
            # Check a complex case of bond charge vsite assignment and overwriting
            (
                Topology.from_molecules(
                    [
                        VirtualSiteMocking.formaldehyde(reverse=False),
                        VirtualSiteMocking.formaldehyde(reverse=True),
                    ],
                ),
                [
                    VirtualSiteMocking.bond_charge_parameter("[O:1]=[*:2]"),
                    VirtualSiteMocking.bond_charge_parameter(
                        "[O:1]=[C:2]",
                        param_multiple=1.5,
                    ),
                    VirtualSiteMocking.bond_charge_parameter(
                        "[O:1]=[CX3:2]",
                        param_multiple=2.0,
                        name="EP2",
                    ),
                ],
                [
                    # charge, sigma, epsilon
                    (0.35 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.7 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.45 * _E, 6.0 * _A, 4.5 * _KJ),  # C=O vsite
                    (-0.6 * _E, 8.0 * _A, 6.0 * _KJ),  # CX3=O vsite
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.0 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.7 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.35 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.45 * _E, 6.0 * _A, 4.5 * _KJ),  # C=O vsite
                    (-0.6 * _E, 8.0 * _A, 6.0 * _KJ),  # CX3=O vsite
                ],
                4,
            ),
            (
                Topology.from_molecules(
                    [
                        VirtualSiteMocking.formaldehyde(reverse=False),
                        VirtualSiteMocking.formaldehyde(reverse=True),
                    ],
                ),
                [VirtualSiteMocking.monovalent_parameter("[O:1]=[C:2]-[H:3]")],
                [
                    # charge, sigma, epsilon
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.4 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.4 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                ],
                4,
            ),
            (
                Topology.from_molecules(
                    [
                        VirtualSiteMocking.hypochlorous_acid(reverse=False),
                        VirtualSiteMocking.hypochlorous_acid(reverse=True),
                    ],
                ),
                [
                    VirtualSiteMocking.divalent_parameter(
                        "[H:2][O:1][Cl:3]",
                        match="all_permutations",
                    ),
                ],
                [
                    # charge, sigma, epsilon
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-0.6 * _E, 4.0 * _A, 5.0 * _KJ),
                ],
                2,
            ),
            (
                Topology.from_molecules(
                    [
                        VirtualSiteMocking.fake_ammonia(reverse=False),
                        VirtualSiteMocking.fake_ammonia(reverse=True),
                    ],
                ),
                [VirtualSiteMocking.trivalent_parameter("[N:1]([Br:2])([Cl:3])[H:4]")],
                [
                    # charge, sigma, epsilon
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.4 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-1.0 * _E, 5.0 * _A, 6.0 * _KJ),
                    (0.4 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.2 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.3 * _E, 10.0 * _A, 0.0 * _KJ),
                    (0.1 * _E, 10.0 * _A, 0.0 * _KJ),
                    (-1.0 * _E, 5.0 * _A, 6.0 * _KJ),
                ],
                2,
            ),
        ],
    )
    def test_create_force(
        self,
        topology: Topology,
        parameters: List[VirtualSiteHandler.VirtualSiteType],
        expected_parameters: List[
            Tuple[openmm_unit.Quantity, openmm_unit.Quantity, openmm_unit.Quantity]
        ],
        expected_n_v_sites: int,
    ):
        expected_n_total = topology.n_atoms + expected_n_v_sites
        # sanity check the test input
        assert len(expected_parameters) == expected_n_total

        handler = VirtualSiteHandler(version="0.3")

        for parameter in parameters:
            handler.add_parameter(parameter=parameter)

        force_field = ForceField()

        force_field.register_parameter_handler(ElectrostaticsHandler(version=0.4))
        force_field.register_parameter_handler(LibraryChargeHandler(version=0.3))
        force_field.get_parameter_handler("LibraryCharges").add_parameter(
            {
                "smirks": "[*:1]",
                "charge": [0.0] * unit.elementary_charge,
            },
        )
        force_field.register_parameter_handler(vdWHandler(version=0.3))
        force_field.get_parameter_handler("vdW").add_parameter(
            {
                "smirks": "[*:1]",
                "epsilon": 0.0 * unit.kilojoule_per_mole,
                "sigma": 1.0 * unit.nanometer,
            },
        )
        force_field.register_parameter_handler(handler)

        system: openmm.System = Interchange.from_smirnoff(
            force_field,
            topology,
        ).to_openmm(
            combine_nonbonded_forces=True,
        )

        assert system.getNumParticles() == expected_n_total

        # TODO: Explicitly ensure virtual sites are collated between moleucles
        #       This is implicitly tested in the construction of the parameter arrays

        assert system.getNumForces() == 1
        force: openmm.NonbondedForce = next(iter(system.getForces()))

        total_charge = 0.0 * openmm_unit.elementary_charge

        for i, (expected_charge, expected_sigma, expected_epsilon) in enumerate(
            expected_parameters,
        ):
            charge, sigma, epsilon = force.getParticleParameters(i)

            # Make sure v-sites are massless.
            assert (
                numpy.isclose(system.getParticleMass(i)._value, 0.0)
            ) == system.isVirtualSite(i)

            assert numpy.isclose(
                expected_charge.m_as(unit.elementary_charge),
                charge.value_in_unit(openmm_unit.elementary_charge),
            )
            assert numpy.isclose(
                expected_sigma.m_as(unit.angstrom),
                sigma.value_in_unit(openmm_unit.angstrom),
            )
            assert numpy.isclose(
                expected_epsilon.m_as(unit.kilojoule_per_mole),
                epsilon.value_in_unit(openmm_unit.kilojoule_per_mole),
            )

            total_charge += charge

        expected_total_charge = sum(
            molecule.total_charge.m_as(unit.elementary_charge)
            for molecule in topology.molecules
        )

        assert numpy.isclose(
            expected_total_charge,
            total_charge.value_in_unit(openmm_unit.elementary_charge),
        )
