import numpy as np
import pytest
from openff.toolkit.tests.test_forcefield import create_ethanol, create_reversed_ethanol
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    ImproperTorsionHandler,
    LibraryChargeHandler,
    ParameterHandler,
    ToolkitAM1BCCHandler,
    UnassignedProperTorsionParameterException,
    UnassignedValenceParameterException,
)
from openff.toolkit.utils import get_data_file_path
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from pydantic import ValidationError
from simtk import openmm
from simtk import unit as simtk_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import OFFBioTop
from openff.interchange.components.smirnoff import (
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFElectrostaticsHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFPotentialHandler,
    SMIRNOFFvdWHandler,
    library_charge_from_molecule,
)
from openff.interchange.drivers.openmm import _get_openmm_energies, get_openmm_energies
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.models import TopologyKey
from openff.interchange.tests import BaseTest
from openff.interchange.utils import get_test_file_path

kcal_mol_a2 = unit.Unit("kilocalorie / (angstrom ** 2 * mole)")
kcal_mol_rad2 = unit.Unit("kilocalorie / (mole * radian ** 2)")


class TestSMIRNOFFPotentialHandler(BaseTest):
    def test_allowed_parameter_handler_types(self):
        class DummyParameterHandler(ParameterHandler):
            pass

        class DummySMIRNOFFHandler(SMIRNOFFPotentialHandler):
            type = "Bonds"
            expression = "1+1"

            @classmethod
            def allowed_parameter_handlers(cls):
                return [DummyParameterHandler]

            @classmethod
            def supported_parameters(cls):
                return list()

        dummy_handler = DummySMIRNOFFHandler()
        angle_Handler = AngleHandler(version=0.3)

        assert DummyParameterHandler in dummy_handler.allowed_parameter_handlers()
        assert AngleHandler not in dummy_handler.allowed_parameter_handlers()
        assert (
            DummyParameterHandler
            not in SMIRNOFFAngleHandler.allowed_parameter_handlers()
        )

        dummy_handler = DummyParameterHandler(version=0.3)

        with pytest.raises(InvalidParameterHandlerError):
            SMIRNOFFAngleHandler._from_toolkit(
                parameter_handler=dummy_handler,
                topology=Topology(),
            )

        with pytest.raises(InvalidParameterHandlerError):
            DummySMIRNOFFHandler._from_toolkit(
                parameter_handler=angle_Handler,
                topology=Topology(),
            )


class TestSMIRNOFFHandlers(BaseTest):
    def test_bond_potential_handler(self):
        top = OFFBioTop.from_molecules(Molecule.from_smiles("O=O"))

        bond_handler = BondHandler(version=0.3)
        bond_parameter = BondHandler.BondType(
            smirks="[*:1]~[*:2]",
            k=1.5 * simtk_unit.kilocalorie_per_mole / simtk_unit.angstrom ** 2,
            length=1.5 * simtk_unit.angstrom,
            id="b1000",
        )
        bond_handler.add_parameter(bond_parameter.to_dict())

        from openff.toolkit.typing.engines.smirnoff import ForceField

        forcefield = ForceField()
        forcefield.register_parameter_handler(bond_handler)
        bond_potentials = SMIRNOFFBondHandler._from_toolkit(
            parameter_handler=forcefield["Bonds"],
            topology=top,
        )

        top_key = TopologyKey(atom_indices=(0, 1))
        pot_key = bond_potentials.slot_map[top_key]
        assert pot_key.associated_handler == "Bonds"
        pot = bond_potentials.potentials[pot_key]

        assert pot.parameters["k"].to(kcal_mol_a2).magnitude == pytest.approx(1.5)

    def test_angle_potential_handler(self):
        top = OFFBioTop.from_molecules(Molecule.from_smiles("CCC"))

        angle_handler = AngleHandler(version=0.3)
        angle_parameter = AngleHandler.AngleType(
            smirks="[*:1]~[*:2]~[*:3]",
            k=2.5 * simtk_unit.kilocalorie_per_mole / simtk_unit.radian ** 2,
            angle=100 * simtk_unit.degree,
            id="b1000",
        )
        angle_handler.add_parameter(angle_parameter.to_dict())

        forcefield = ForceField()
        forcefield.register_parameter_handler(angle_handler)
        angle_potentials = SMIRNOFFAngleHandler._from_toolkit(
            parameter_handler=forcefield["Angles"],
            topology=top,
        )

        top_key = TopologyKey(atom_indices=(0, 1, 2))
        pot_key = angle_potentials.slot_map[top_key]
        assert pot_key.associated_handler == "Angles"
        pot = angle_potentials.potentials[pot_key]

        assert pot.parameters["k"].to(kcal_mol_rad2).magnitude == pytest.approx(2.5)

    def test_store_improper_torsion_matches(self):
        formaldehyde: Molecule = Molecule.from_mapped_smiles("[H:3][C:1]([H:4])=[O:2]")

        parameter_handler = ImproperTorsionHandler(version=0.3)
        parameter_handler.add_parameter(
            parameter=ImproperTorsionHandler.ImproperTorsionType(
                smirks="[*:1]~[#6X3:2](~[*:3])~[*:4]",
                periodicity1=2,
                phase1=180.0 * simtk_unit.degree,
                k1=1.1 * simtk_unit.kilocalorie_per_mole,
            )
        )

        potential_handler = SMIRNOFFImproperTorsionHandler()
        potential_handler.store_matches(parameter_handler, formaldehyde.to_topology())

        assert len(potential_handler.slot_map) == 3

        assert (
            TopologyKey(atom_indices=(0, 1, 2, 3), mult=0) in potential_handler.slot_map
        )
        assert (
            TopologyKey(atom_indices=(0, 2, 3, 1), mult=0) in potential_handler.slot_map
        )
        assert (
            TopologyKey(atom_indices=(0, 3, 1, 2), mult=0) in potential_handler.slot_map
        )

    def test_electrostatics_am1_handler(self):
        molecule = Molecule.from_smiles("C")
        molecule.assign_partial_charges(partial_charge_method="am1bcc")

        # Explicitly store these, since results differ RDKit/AmberTools vs. OpenEye
        reference_charges = [c._value for c in molecule.partial_charges]

        top = OFFBioTop.from_molecules(molecule)

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            ToolkitAM1BCCHandler(version=0.3),
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsHandler._from_toolkit(
            parameter_handlers, top
        )
        np.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler.charges.values()],
            reference_charges,
        )

    def test_electrostatics_library_charges(self):
        top = OFFBioTop.from_molecules(Molecule.from_smiles("C"))

        library_charge_handler = LibraryChargeHandler(version=0.3)
        library_charge_handler.add_parameter(
            {
                "smirks": "[#6X4:1]-[#1:2]",
                "charge1": -0.1 * simtk_unit.elementary_charge,
                "charge2": 0.025 * simtk_unit.elementary_charge,
            }
        )

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            library_charge_handler,
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsHandler._from_toolkit(
            parameter_handlers, top
        )

        np.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler.charges.values()],
            [-0.1, 0.025, 0.025, 0.025, 0.025],
        )

    def test_electrostatics_charge_increments(self):
        molecule = Molecule.from_mapped_smiles("[Cl:1][H:2]")
        top = OFFBioTop.from_molecules(molecule)

        molecule.assign_partial_charges(partial_charge_method="am1-mulliken")

        reference_charges = [c._value for c in molecule.partial_charges]
        reference_charges[0] += 0.1
        reference_charges[1] -= 0.1

        charge_increment_handler = ChargeIncrementModelHandler(version=0.3)
        charge_increment_handler.add_parameter(
            {
                "smirks": "[#17:1]-[#1:2]",
                "charge_increment1": 0.1 * simtk_unit.elementary_charge,
                "charge_increment2": -0.1 * simtk_unit.elementary_charge,
            }
        )

        parameter_handlers = [
            ElectrostaticsHandler(version=0.3),
            charge_increment_handler,
        ]

        electrostatics_handler = SMIRNOFFElectrostaticsHandler._from_toolkit(
            parameter_handlers, top
        )

        # AM1-Mulliken charges are [-0.168,  0.168], increments are [0.1, -0.1],
        # sum is [-0.068,  0.068]
        np.testing.assert_allclose(
            [charge.m_as(unit.e) for charge in electrostatics_handler.charges.values()],
            reference_charges,
        )


class TestUnassignedParameters(BaseTest):
    def test_catch_unassigned_bonds(self, parsley, ethanol_top):
        for param in parsley["Bonds"].parameters:
            param.smirks = "[#99:1]-[#99:2]"

        parsley.deregister_parameter_handler(parsley["Constraints"])

        with pytest.raises(
            UnassignedValenceParameterException,
            match="BondHandler was not able to find par",
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=ethanol_top)

    def test_catch_unassigned_angles(self, parsley, ethanol_top):
        for param in parsley["Angles"].parameters:
            param.smirks = "[#99:1]-[#99:2]-[#99:3]"

        with pytest.raises(
            UnassignedValenceParameterException,
            match="AngleHandler was not able to find par",
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=ethanol_top)

    def test_catch_unassigned_torsions(self, parsley, ethanol_top):
        for param in parsley["ProperTorsions"].parameters:
            param.smirks = "[#99:1]-[#99:2]-[#99:3]-[#99:4]"

        with pytest.raises(
            UnassignedProperTorsionParameterException,
            match="- Topology indices [(]5, 0, 1, 6[)]: "
            r"names and elements [(](H\d+)? H[)], [(](C\d+)? C[)], [(](C\d+)? C[)], [(](H\d+)? H[)],",
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=ethanol_top)


class TestConstraints:
    @pytest.mark.parametrize(
        ("mol", "n_constraints"),
        [
            ("C", 4),
            ("CC", 6),
        ],
    )
    def test_num_constraints(self, mol, n_constraints):
        force_field = ForceField("openff-1.0.0.offxml")

        bond_handler = force_field["Bonds"]
        constraint_handler = force_field["Constraints"]

        topology = Molecule.from_smiles(mol).to_topology()

        constraints = SMIRNOFFConstraintHandler._from_toolkit(
            parameter_handler=[
                val for val in [bond_handler, constraint_handler] if val is not None
            ],
            topology=topology,
        )

        assert len(constraints.slot_map) == n_constraints


def test_library_charges_from_molecule():
    mol = Molecule.from_mapped_smiles("[Cl:1][C:2]#[C:3][F:4]")

    with pytest.raises(ValueError, match="missing partial"):
        library_charge_from_molecule(mol)

    mol.partial_charges = np.linspace(-0.3, 0.3, 4) * simtk_unit.elementary_charge

    library_charges = library_charge_from_molecule(mol)

    assert isinstance(library_charges, LibraryChargeHandler.LibraryChargeType)
    assert library_charges.smirks == mol.to_smiles(mapped=True)
    assert library_charges.charge == [*mol.partial_charges]


class TestBondOrderInterpolation(BaseTest):
    xml_ff_bo_bonds = """<?xml version='1.0' encoding='ASCII'?>
    <SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
      <Bonds version="0.3" fractional_bondorder_method="AM1-Wiberg" fractional_bondorder_interpolation="linear">
        <Bond smirks="[#6:1]~[#8:2]" id="bbo1"
            k_bondorder1="100.0 * kilocalories_per_mole/angstrom**2"
            k_bondorder2="1000.0 * kilocalories_per_mole/angstrom**2"
            length_bondorder1="1.5 * angstrom"
            length_bondorder2="1.0 * angstrom"/>
      </Bonds>
    </SMIRNOFF>
    """

    @pytest.mark.slow()
    def test_input_bond_orders_ignored(self):
        """Test that conformers existing in the topology are not considered in the bond order interpolation
        part of the parametrization process"""
        from openff.toolkit.tests.test_forcefield import create_ethanol

        mol = create_ethanol()
        mol.assign_fractional_bond_orders(bond_order_model="am1-wiberg")
        mod_mol = Molecule(mol)
        for bond in mod_mol.bonds:
            bond.fractional_bond_order += 0.1

        top = Topology.from_molecules(mol)
        mod_top = Topology.from_molecules(mod_mol)

        forcefield = ForceField(
            get_data_file_path("test_forcefields/test_forcefield.offxml"),
            self.xml_ff_bo_bonds,
        )

        bonds = SMIRNOFFBondHandler._from_toolkit(
            parameter_handler=forcefield["Bonds"], topology=top
        )
        bonds_mod = SMIRNOFFBondHandler._from_toolkit(
            parameter_handler=forcefield["Bonds"], topology=mod_top
        )

        for pot_key1, pot_key2 in zip(
            bonds.slot_map.values(), bonds_mod.slot_map.values()
        ):
            k1 = bonds.potentials[pot_key1].parameters["k"]
            k2 = bonds_mod.potentials[pot_key2].parameters["k"]
            assert k1 == k2

    def test_input_conformers_ignored(self):
        """Test that conformers existing in the topology are not considered in the bond order interpolation
        part of the parametrization process"""
        from openff.toolkit.tests.test_forcefield import create_ethanol

        mol = create_ethanol()
        mol.assign_fractional_bond_orders(bond_order_model="am1-wiberg")
        mod_mol = Molecule(mol)
        mod_mol.generate_conformers()
        tmp = mod_mol._conformers[0][0][0]
        mod_mol._conformers[0][0][0] = mod_mol._conformers[0][1][0]
        mod_mol._conformers[0][1][0] = tmp

        top = Topology.from_molecules(mol)
        mod_top = Topology.from_molecules(mod_mol)

        forcefield = ForceField(
            get_data_file_path("test_forcefields/test_forcefield.offxml"),
            self.xml_ff_bo_bonds,
        )

        bonds = SMIRNOFFBondHandler._from_toolkit(
            parameter_handler=forcefield["Bonds"], topology=top
        )
        bonds_mod = SMIRNOFFBondHandler._from_toolkit(
            parameter_handler=forcefield["Bonds"], topology=mod_top
        )

        for key1, key2 in zip(bonds.potentials, bonds_mod.potentials):
            k1 = bonds.potentials[key1].parameters["k"]
            k2 = bonds_mod.potentials[key2].parameters["k"]
            assert k1 == k2

    @pytest.mark.slow()
    def test_basic_bond_order_interpolation_energies(self):

        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml",
            self.xml_ff_bo_bonds,
        )

        mol = Molecule.from_file(get_data_file_path("molecules/CID20742535_anion.sdf"))
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()

        out = Interchange.from_smirnoff(forcefield, top)
        out.box = [4, 4, 4] * unit.nanometer
        out.positions = mol.conformers[0]

        interchange_bond_energy = get_openmm_energies(
            out, combine_nonbonded_forces=True
        ).energies["Bond"]
        toolkit_bond_energy = _get_openmm_energies(
            forcefield.create_openmm_system(top),
            box_vectors=[[4, 0, 0], [0, 4, 0], [0, 0, 4]] * simtk_unit.nanometer,
            positions=mol.conformers[0],
        ).energies["Bond"]

        assert abs(interchange_bond_energy - toolkit_bond_energy).m < 1e-2

        new = out.to_openmm(combine_nonbonded_forces=True)
        ref = forcefield.create_openmm_system(top)

        new_k = []
        new_length = []
        for force in new.getForces():
            if type(force) == openmm.HarmonicBondForce:
                for i in range(force.getNumBonds()):
                    new_k.append(force.getBondParameters(i)[3]._value)
                    new_length.append(force.getBondParameters(i)[2]._value)

        ref_k = []
        ref_length = []
        for force in ref.getForces():
            if type(force) == openmm.HarmonicBondForce:
                for i in range(force.getNumBonds()):
                    ref_k.append(force.getBondParameters(i)[3]._value)
                    ref_length.append(force.getBondParameters(i)[2]._value)

        np.testing.assert_allclose(ref_k, new_k, rtol=3e-5)

    def test_fractional_bondorder_invalid_interpolation_method(self):
        """
        Ensure that requesting an invalid interpolation method leads to a
        FractionalBondOrderInterpolationMethodUnsupportedError
        """
        mol = Molecule.from_smiles("CCO")

        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml", self.xml_ff_bo_bonds
        )
        forcefield.get_parameter_handler(
            "ProperTorsions"
        )._fractional_bondorder_interpolation = "invalid method name"
        topology = Topology.from_molecules([mol])

        # TODO: Make this a more descriptive custom exception
        with pytest.raises(ValidationError):
            Interchange.from_smirnoff(forcefield, topology)


@skip_if_missing("jax")
class TestMatrixRepresentations(BaseTest):
    @pytest.mark.parametrize(
        ("handler_name", "n_ff_terms", "n_sys_terms"),
        [("vdW", 10, 72), ("Bonds", 8, 64), ("Angles", 6, 104)],
    )
    def test_to_force_field_to_system_parameters(
        self, parsley, ethanol_top, handler_name, n_ff_terms, n_sys_terms
    ):
        import jax

        if handler_name == "Bonds":
            handler = SMIRNOFFBondHandler._from_toolkit(
                parameter_handler=parsley["Bonds"],
                topology=ethanol_top,
            )
        elif handler_name == "Angles":
            handler = SMIRNOFFAngleHandler._from_toolkit(
                parameter_handler=parsley[handler_name],
                topology=ethanol_top,
            )
        elif handler_name == "vdW":
            handler = SMIRNOFFvdWHandler._from_toolkit(
                parameter_handler=parsley[handler_name],
                topology=ethanol_top,
            )
        else:
            raise NotImplementedError()

        p = handler.get_force_field_parameters()

        assert isinstance(p, jax.interpreters.xla.DeviceArray)
        assert np.prod(p.shape) == n_ff_terms

        q = handler.get_system_parameters()

        assert isinstance(q, jax.interpreters.xla.DeviceArray)
        assert np.prod(q.shape) == n_sys_terms

        assert jax.numpy.allclose(q, handler.parametrize(p))

        param_matrix = handler.get_param_matrix()

        ref_file = get_test_file_path(f"ethanol_param_{handler_name.lower()}.npy")
        ref = jax.numpy.load(ref_file)

        assert jax.numpy.allclose(ref, param_matrix)

        # TODO: Update with other handlers that can safely be assumed to follow 1:1 slot:smirks mapping
        if handler_name in ["vdW", "Bonds", "Angles"]:
            assert np.allclose(
                np.sum(param_matrix, axis=1), np.ones(param_matrix.shape[0])
            )


class TestParameterInterpolation(BaseTest):
    xml_ff_bo = """<?xml version='1.0' encoding='ASCII'?>
    <SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
      <Bonds version="0.3" fractional_bondorder_method="AM1-Wiberg"
        fractional_bondorder_interpolation="linear">
        <Bond
          smirks="[#6X4:1]~[#8X2:2]"
          id="bbo1"
          k_bondorder1="101.0 * kilocalories_per_mole/angstrom**2"
          k_bondorder2="123.0 * kilocalories_per_mole/angstrom**2"
          length_bondorder1="1.4 * angstrom"
          length_bondorder2="1.3 * angstrom"
          />
      </Bonds>
      <ProperTorsions version="0.3" potential="k*(1+cos(periodicity*theta-phase))">
        <Proper smirks="[*:1]~[#6X3:2]~[#6X3:3]~[*:4]" id="tbo1" periodicity1="2" phase1="0.0 * degree"
        k1_bondorder1="1.00*kilocalories_per_mole" k1_bondorder2="1.80*kilocalories_per_mole" idivf1="1.0"/>
        <Proper smirks="[*:1]~[#6X4:2]~[#8X2:3]~[*:4]" id="tbo2" periodicity1="2" phase1="0.0 * degree"
        k1_bondorder1="1.00*kilocalories_per_mole" k1_bondorder2="1.80*kilocalories_per_mole" idivf1="1.0"/>
      </ProperTorsions>
    </SMIRNOFF>
    """

    @pytest.mark.xfail(reason="Not yet implemented using input bond orders")
    def test_bond_order_interpolation(self):
        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml", self.xml_ff_bo
        )

        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)

        mol.bonds[1].fractional_bond_order = 1.5

        top = mol.to_topology()

        out = Interchange.from_smirnoff(forcefield, mol.to_topology())

        top_key = TopologyKey(
            atom_indices=(1, 2),
            bond_order=top.get_bond_between(1, 2).bond.fractional_bond_order,
        )
        assert out["Bonds"].potentials[out["Bonds"].slot_map[top_key]].parameters[
            "k"
        ] == 300 * unit.Unit("kilocalories / mol / angstrom ** 2")

    @pytest.mark.slow()
    @pytest.mark.xfail(reason="Not yet implemented using input bond orders")
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

        top = mol.to_topology()

        out = Interchange.from_smirnoff(forcefield, top)

        bond1_top_key = TopologyKey(
            atom_indices=(2, 3),
            bond_order=top.get_bond_between(2, 3).bond.fractional_bond_order,
        )
        bond1_pot_key = out["Bonds"].slot_map[bond1_top_key]

        bond2_top_key = TopologyKey(
            atom_indices=(0, 4),
            bond_order=top.get_bond_between(0, 4).bond.fractional_bond_order,
        )
        bond2_pot_key = out["Bonds"].slot_map[bond2_top_key]

        assert np.allclose(
            out["Bonds"].potentials[bond1_pot_key].parameters["k"],
            300.0 * unit.Unit("kilocalories / mol / angstrom ** 2"),
        )

        assert np.allclose(
            out["Bonds"].potentials[bond2_pot_key].parameters["k"],
            180.0 * unit.Unit("kilocalories / mol / angstrom ** 2"),
        )

    @pytest.mark.parametrize(
        (
            "get_molecule",
            "k_torsion_interpolated",
            "k_bond_interpolated",
            "length_bond_interpolated",
            "central_atoms",
        ),
        [
            (create_ethanol, 4.16586914, 42208.5402, 0.140054167256, (1, 2)),
            (create_reversed_ethanol, 4.16586914, 42208.5402, 0.140054167256, (7, 6)),
        ],
    )
    def test_fractional_bondorder_from_molecule(
        self,
        get_molecule,
        k_torsion_interpolated,
        k_bond_interpolated,
        length_bond_interpolated,
        central_atoms,
    ):
        """Copied from the toolkit with modified reference constants.
        Force constant computed by interpolating (k1, k2) = (101, 123) kcal/A**2/mol
        with bond order 1.00093035 (AmberTools 21.4, Python 3.8, macOS):
            101 + (123 - 101) * (0.00093035) = 101.0204677 kcal/A**2/mol
            = 42266.9637 kJ/nm**2/mol

        Same process with bond length (1.4, 1.3) A gives 0.1399906965 nm
        Same process with torsion k (1.0, 1.8) kcal/mol gives 4.18711406752 kJ/mol

        Using OpenEye (openeye-toolkits 2021.1.1, Python 3.8, macOS):
            bond order 0.9945832743790813
            bond k = 42208.5402 kJ/nm**2/mol
            bond length = 0.14005416725620918 nm
            torsion k = 4.16586914 kilojoules kJ/mol

        """
        mol = get_molecule()
        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml", self.xml_ff_bo
        )
        topology = Topology.from_molecules(mol)

        out = Interchange.from_smirnoff(forcefield, topology)
        out.box = [4, 4, 4]
        omm_system = out.to_openmm(combine_nonbonded_forces=True)

        # Verify that the assigned bond parameters were correctly interpolated
        off_bond_force = [
            force
            for force in omm_system.getForces()
            if isinstance(force, openmm.HarmonicBondForce)
        ][0]

        for idx in range(off_bond_force.getNumBonds()):
            params = off_bond_force.getBondParameters(idx)

            atom1, atom2 = params[0], params[1]
            atom1_mol, atom2_mol = central_atoms

            if ((atom1 == atom1_mol) and (atom2 == atom2_mol)) or (
                (atom1 == atom2_mol) and (atom2 == atom1_mol)
            ):
                k = params[-1]
                length = params[-2]
                np.testing.assert_allclose(
                    k / k.unit, k_bond_interpolated, atol=0, rtol=2e-6
                )
                np.testing.assert_allclose(
                    length / length.unit,
                    length_bond_interpolated,
                    atol=0,
                    rtol=2e-6,
                )

        # Verify that the assigned torsion parameters were correctly interpolated
        off_torsion_force = [
            force
            for force in omm_system.getForces()
            if isinstance(force, openmm.PeriodicTorsionForce)
        ][0]

        for idx in range(off_torsion_force.getNumTorsions()):
            params = off_torsion_force.getTorsionParameters(idx)

            atom2, atom3 = params[1], params[2]
            atom2_mol, atom3_mol = central_atoms

            if ((atom2 == atom2_mol) and (atom3 == atom3_mol)) or (
                (atom2 == atom3_mol) and (atom3 == atom2_mol)
            ):
                k = params[-1]
                np.testing.assert_allclose(
                    k / k.unit, k_torsion_interpolated, atol=0, rtol=2e-6
                )
