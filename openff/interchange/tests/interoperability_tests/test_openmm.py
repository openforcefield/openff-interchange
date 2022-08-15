import numpy
import openmm
import pytest
from openff.toolkit.tests.test_forcefield import create_ethanol
from openff.toolkit.tests.utils import compare_system_parameters, get_data_file_path
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler
from openff.units import unit
from openmm import app
from openmm import unit as openmm_unit

from openff.interchange import Interchange
from openff.interchange.components.smirnoff import SMIRNOFFVirtualSiteHandler
from openff.interchange.drivers.openmm import _get_openmm_energies, get_openmm_energies
from openff.interchange.exceptions import (
    MissingPositionsError,
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.interchange.interop.openmm import from_openmm, to_openmm_topology
from openff.interchange.tests import _BaseTest, get_test_file_path

# WISHLIST: Add tests for reaction-field if implemented

nonbonded_methods = [
    {
        "vdw_method": "cutoff",
        "electrostatics_periodic": "PME",
        "periodic": True,
        "result": openmm.NonbondedForce.PME,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_periodic": "PME",
        "periodic": False,
        "result": openmm.NonbondedForce.NoCutoff,
    },
    {
        "vdw_method": "PME",
        "electrostatics_periodic": "PME",
        "periodic": True,
        "result": openmm.NonbondedForce.LJPME,
    },
    {
        "vdw_method": "PME",
        "electrostatics_periodic": "PME",
        "periodic": False,
        "result": UnsupportedCutoffMethodError,
    },
]


def _get_num_virtual_sites(openmm_topology: app.Topology) -> int:
    return sum(atom.element is None for atom in openmm_topology.atoms())


def _compare_openmm_topologies(top1: app.Topology, top2: app.Topology):
    """
    In lieu of first-class serializaiton in OpenMM (https://github.com/openmm/openmm/issues/1543),
    do some quick heuristics to roughly compare two OpenMM Topology objects.
    """
    for method_name in ["getNumAtoms", "getNumBonds", "getNumChains", "getNumResidues"]:
        assert getattr(top1, method_name)() == getattr(top2, method_name)()

    assert (top1.getPeriodicBoxVectors() == top2.getPeriodicBoxVectors()).all()


class TestOpenMM(_BaseTest):
    @pytest.mark.parametrize("inputs", nonbonded_methods)
    def test_openmm_nonbonded_methods(self, inputs):
        """See test_nonbonded_method_resolution in openff/toolkit/tests/test_forcefield.py"""
        vdw_method = inputs["vdw_method"]
        electrostatics_method = inputs["electrostatics_periodic"]
        periodic = inputs["periodic"]
        result = inputs["result"]

        molecules = [create_ethanol()]
        forcefield = ForceField("test_forcefields/test_forcefield.offxml")

        pdbfile = app.PDBFile(get_data_file_path("systems/test_systems/1_ethanol.pdb"))
        topology = Topology.from_openmm(pdbfile.topology, unique_molecules=molecules)

        if not periodic:
            topology.box_vectors = None

        forcefield.get_parameter_handler("vdW", {}).method = vdw_method
        forcefield.get_parameter_handler(
            "Electrostatics", {}
        ).periodic_potential = electrostatics_method
        interchange = Interchange.from_smirnoff(
            force_field=forcefield, topology=topology
        )
        if type(result) == int:
            nonbonded_method = result
            # The method is validated and may raise an exception if it's not supported.
            forcefield.get_parameter_handler("vdW", {}).method = vdw_method
            forcefield.get_parameter_handler(
                "Electrostatics", {}
            ).periodic_potential = electrostatics_method
            interchange = Interchange.from_smirnoff(
                force_field=forcefield, topology=topology
            )
            openmm_system = interchange.to_openmm(combine_nonbonded_forces=True)
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    assert force.getNonbondedMethod() == nonbonded_method
                    break
            else:
                raise Exception
        elif issubclass(result, (BaseException, Exception)):
            exception = result
            with pytest.raises(exception):
                interchange.to_openmm(combine_nonbonded_forces=True)
        else:
            raise Exception("uh oh")

    def test_unsupported_mixing_rule(self):
        molecules = [create_ethanol()]
        pdbfile = app.PDBFile(get_data_file_path("systems/test_systems/1_ethanol.pdb"))
        topology = Topology.from_openmm(pdbfile.topology, unique_molecules=molecules)

        forcefield = ForceField("test_forcefields/test_forcefield.offxml")
        openff_sys = Interchange.from_smirnoff(
            force_field=forcefield, topology=topology
        )

        openff_sys["vdW"].mixing_rule = "geometric"

        with pytest.raises(UnsupportedExportError, match="default NonbondedForce"):
            openff_sys.to_openmm(combine_nonbonded_forces=True)

    @pytest.mark.xfail(reason="Broken because of splitting non-bonded forces")
    @pytest.mark.slow()
    @pytest.mark.parametrize("n_mols", [1, 2])
    @pytest.mark.parametrize(
        "mol",
        [
            "C",
            "CC",  # Adds a proper torsion term(s)
            "OC=O",  # Simplest molecule with a multi-term torsion
            "CCOC",  # This hits t86, which has a non-1.0 idivf
            "C1COC(=O)O1",  # This adds an improper, i2
        ],
    )
    def test_from_openmm_single_mols(sage, mol, n_mols):
        """
        Test that ForceField.create_openmm_system and Interchange.to_openmm produce
        objects with similar energies

        TODO: Tighten tolerances
        TODO: Test periodic and non-periodic
        """
        mol = Molecule.from_smiles(mol)
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules(n_mols * [mol])
        mol.conformers[0] -= numpy.min(mol.conformers)

        top.box_vectors = 15 * numpy.eye(3) * unit.nanometer

        if n_mols == 1:
            positions = mol.conformers[0]
        elif n_mols == 2:
            positions = numpy.concatenate(
                [mol.conformers[0], mol.conformers[0] + 3 * mol.conformers[0].units]
            )

        toolkit_system = sage.create_openmm_system(top)

        native_system = Interchange.from_smirnoff(
            force_field=sage, topology=top
        ).to_openmm()

        toolkit_energy = _get_openmm_energies(
            omm_sys=toolkit_system,
            box_vectors=toolkit_system.getDefaultPeriodicBoxVectors(),
            positions=positions,
        )
        native_energy = _get_openmm_energies(
            omm_sys=native_system,
            box_vectors=native_system.getDefaultPeriodicBoxVectors(),
            positions=positions,
        )

        toolkit_energy.compare(native_energy)

    @pytest.mark.xfail(reason="Broken because of splitting non-bonded forces")
    @pytest.mark.slow()
    @pytest.mark.parametrize("mol_smi", ["C", "CC", "CCO"])
    def test_openmm_roundtrip(self, sage, mol_smi):
        mol = Molecule.from_smiles(mol_smi)
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()
        omm_top = top.to_openmm()

        off_sys = Interchange.from_smirnoff(sage, top)

        off_sys.box = [4, 4, 4]
        off_sys.positions = mol.conformers[0].value_in_unit(openmm_unit.nanometer)

        omm_sys = off_sys.to_openmm(combine_nonbonded_forces=True)

        converted = from_openmm(
            topology=omm_top,
            system=omm_sys,
        )

        converted.box = off_sys.box
        converted.positions = off_sys.positions

        get_openmm_energies(off_sys).compare(
            get_openmm_energies(converted, combine_nonbonded_forces=True),
        )

    @pytest.mark.xfail(reason="Broken because of splitting non-bonded forces")
    @pytest.mark.slow()
    def test_combine_nonbonded_forces(self, sage):

        mol = Molecule.from_smiles("ClC#CCl")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(force_field=sage, topology=mol.to_topology())
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]

        num_forces_combined = out.to_openmm(
            combine_nonbonded_forces=True
        ).getNumForces()
        num_forces_uncombined = out.to_openmm(
            combine_nonbonded_forces=False
        ).getNumForces()

        # The "new" forces are the split-off vdW forces, the 1-4 vdW, and the 1-4 electrostatics
        assert num_forces_combined + 3 == num_forces_uncombined

        separate = get_openmm_energies(out, combine_nonbonded_forces=False)
        combined = get_openmm_energies(out, combine_nonbonded_forces=True)

        assert (
            separate["vdW"] + separate["Electrostatics"] - combined["Nonbonded"]
        ).m < 0.001

    def test_openmm_no_angle_force_if_constrained(self):
        # Sage includes angle parameters for water and also TIP3P constraints
        tip3p = ForceField("openff-2.0.0.offxml")

        topology = Molecule.from_smiles("O").to_topology()
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        interchange = Interchange.from_smirnoff(tip3p, topology)
        openmm_system = interchange.to_openmm(combine_nonbonded_forces=True)

        # The only angle in the system (H-O-H) includes bonds with constrained lengths
        # and a constrained angle, so by convention a force should NOT be added
        for force in openmm_system.getForces():
            if type(force) == openmm.HarmonicAngleForce:
                assert force.getNumAngles() == 0
                break
        else:
            raise Exception("No HarmonicAngleForce found")

    def test_nonharmonic_angle(self, sage, ethanol_top):
        out = Interchange.from_smirnoff(sage, ethanol_top)
        out["Angles"].expression = "k/2*(cos(theta)-cos(angle))**2"

        system = out.to_openmm()

        def _is_custom_angle(force):
            return isinstance(force, openmm.CustomAngleForce)

        assert len([f for f in system.getForces() if _is_custom_angle(f)]) == 1

        for force in system.getForces():
            if _is_custom_angle(force):
                assert force.getEnergyFunction() == "k/2*(cos(theta)-cos(angle))^2"

    def test_openmm_no_valence_forces_with_no_handler(self, sage):
        ethanol = create_ethanol()

        original_system = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True
        )
        assert original_system.getNumForces() == 4

        sage.deregister_parameter_handler("Constraints")
        sage.deregister_parameter_handler("Bonds")

        no_bonds = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True
        )
        assert no_bonds.getNumForces() == 3

        sage.deregister_parameter_handler("Angles")

        no_angles = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True
        )
        assert no_angles.getNumForces() == 2

    def test_openmm_only_electrostatics_no_vdw(self):
        force_field_only_charges = ForceField(get_test_file_path("no_vdw.offxml"))
        molecule = Molecule.from_smiles("[H][Cl]")

        system = Interchange.from_smirnoff(
            force_field_only_charges, [molecule]
        ).to_openmm(
            combine_nonbonded_forces=True,
        )

        assert system.getForce(0).getParticleParameters(0)[0]._value == 1.0
        assert system.getForce(0).getParticleParameters(1)[0]._value == -1.0

    def test_nonstandard_cutoffs_match(self):
        """Test that multiple nonbonded forces use the same cutoff."""
        force_field = ForceField("test_forcefields/test_forcefield.offxml")
        topology = Molecule.from_smiles("C").to_topology()
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        cutoff = unit.Quantity(1.555, unit.nanometer)

        force_field["vdW"].cutoff = cutoff

        interchange = Interchange.from_smirnoff(
            force_field=force_field,
            topology=topology,
        )

        system = interchange.to_openmm(combine_nonbonded_forces=False)

        # For now, just make sure all non-bonded forces use the vdW handler's cutoff
        for force in system.getForces():
            if type(force) in (openmm.NonbondedForce, openmm.CustomNonbondedForce):
                assert force.getCutoffDistance().value_in_unit(
                    openmm_unit.nanometer
                ) == pytest.approx(cutoff.m_as(unit.nanometer))


class TestOpenMMSwitchingFunction(_BaseTest):
    def test_switching_function_applied(self, sage, basic_top):
        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert force.getUseSwitchingFunction()
                assert force.getSwitchingDistance().value_in_unit(
                    openmm_unit.angstrom
                ) == pytest.approx(8), force.getSwitchingDistance()

        assert found_force, "NonbondedForce not found in system"

    def test_switching_function_not_applied(self, sage, basic_top):
        sage["vdW"].switch_width = 0.0 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert not force.getUseSwitchingFunction()
                assert force.getSwitchingDistance() == -1 * openmm_unit.nanometer

        assert found_force, "NonbondedForce not found in system"

    def test_switching_function_nonstandard(self, sage, basic_top):
        sage["vdW"].switch_width = 0.12345 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert force.getUseSwitchingFunction()
                assert (
                    force.getSwitchingDistance() - (9 - 0.12345) * openmm_unit.angstrom
                ) < 1e-10 * openmm_unit.angstrom

        assert found_force, "NonbondedForce not found in system"


@pytest.mark.slow()
class TestOpenMMVirtualSites(_BaseTest):
    @pytest.fixture()
    def sage_with_sigma_hole(self, sage):
        """Fixture that loads an SMIRNOFF XML with a C-Cl sigma hole."""
        # TODO: Move this into BaseTest to that GROMACS and others can access it
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        sigma_type = VirtualSiteHandler.VirtualSiteType(
            name="EP",
            smirks="[#6:1]-[#17:2]",
            distance=1.4 * unit.angstrom,
            type="BondCharge",
            match="once",
            charge_increment1=0.1 * unit.elementary_charge,
            charge_increment2=0.2 * unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=sigma_type)
        sage.register_parameter_handler(virtual_site_handler)

        return sage

    @pytest.fixture()
    def sage_with_monovalent_lone_pair(self, sage):
        """Fixture that loads an SMIRNOFF XML for argon"""
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        carbonyl_type = VirtualSiteHandler.VirtualSiteMonovalentLonePairType(
            name="EP",
            smirks="[O:1]=[C:2]-[C:3]",
            distance=0.3 * unit.angstrom,
            type="MonovalentLonePair",
            match="once",
            outOfPlaneAngle=0.0 * unit.degree,
            inPlaneAngle=120.0 * unit.degree,
            charge_increment1=0.05 * unit.elementary_charge,
            charge_increment2=0.1 * unit.elementary_charge,
            charge_increment3=0.15 * unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=carbonyl_type)
        sage.register_parameter_handler(virtual_site_handler)

        return sage

    @pytest.mark.skip(reason="Virtual sites not supported")
    def test_sigma_hole_example(self, sage_with_sigma_hole):
        """Test that a single-molecule sigma hole example runs"""
        mol = Molecule.from_smiles("CCl")
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=sage_with_sigma_hole, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=sage_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=sage_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=sage_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_openmm_energies(out, combine_nonbonded_forces=True)

        compare_system_parameters(
            out.to_openmm(combine_nonbonded_forces=True),
            sage_with_sigma_hole.create_openmm_system(mol.to_topology()),
        )
        """
        import numpy
        import parmed

        out.to_top("sigma.top")
        gmx_top = parmed.load_file("sigma.top")

        assert abs(numpy.sum([p.charge for p in gmx_top.atoms])) < 1e-3
        """

    @pytest.mark.skip(reason="Virtual sites not supported")
    def test_carbonyl_example(self, sage_with_monovalent_lone_pair):
        """Test that a single-molecule DivalentLonePair example runs"""
        mol = Molecule.from_smiles("CC=O")
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=sage_with_monovalent_lone_pair, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=sage_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=sage_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=sage_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_openmm_energies(out, combine_nonbonded_forces=True)

        compare_system_parameters(
            out.to_openmm(combine_nonbonded_forces=True),
            sage_with_monovalent_lone_pair.create_openmm_system(mol.to_topology()),
        )

    @pytest.mark.skip(reason="virtual sites in development")
    def test_tip5p_num_exceptions(self):
        tip5p = ForceField(get_test_file_path("tip5p.offxml"))
        water = Molecule.from_smiles("O")
        water.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(tip5p, [water]).to_openmm(
            combine_nonbonded_forces=True
        )

        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getNumExceptions() == 12


class TestToOpenMMTopology(_BaseTest):
    def test_num_virtual_sites(self):
        from openff.interchange.interop.openmm import to_openmm_topology

        tip4p = ForceField("openff-2.0.0.offxml", get_test_file_path("tip4p.offxml"))
        water = Molecule.from_smiles("O")

        out = Interchange.from_smirnoff(tip4p, [water])

        assert _get_num_virtual_sites(to_openmm_topology(out)) == 1

        # TODO: Monkeypatch Topology.to_openmm() and emit a warning when it seems
        #       to be used while virtual sites are present in a handler
        assert _get_num_virtual_sites(out.topology.to_openmm()) == 0

    def test_interchange_method(self):
        """
        Ensure similar-ish behavior between `to_openmm_topology` as a standalone function
        and as the wrapped method of the same name on the `Interchange` class.
        """
        tip4p = ForceField("openff-2.0.0.offxml", get_test_file_path("tip4p.offxml"))
        topology = Molecule.from_smiles("O").to_topology()
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        out = Interchange.from_smirnoff(tip4p, topology)

        _compare_openmm_topologies(out.to_openmm_topology(), to_openmm_topology(out))


class TestToOpenMMPositions(_BaseTest):
    @pytest.mark.parametrize("include_virtual_sites", [True, False])
    def test_positions(self, include_virtual_sites):
        from openff.interchange.interop.openmm import to_openmm_positions

        tip4p = ForceField("openff-2.0.0.offxml", get_test_file_path("tip4p.offxml"))
        water = Molecule.from_smiles("O")
        water.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(tip4p, [water])

        positions = to_openmm_positions(
            out, include_virtual_sites=include_virtual_sites
        )

        assert positions.shape == (4, 3) if include_virtual_sites else (3, 3)

        numpy.testing.assert_allclose(
            positions.to(unit.angstrom)[:3], water.conformers[0]
        )


class TestOpenMMToPDB(_BaseTest):
    def test_to_pdb(self, sage):
        import mdtraj as md

        molecule = Molecule.from_smiles("O")

        out = Interchange.from_smirnoff(sage, molecule.to_topology())

        with pytest.raises(MissingPositionsError):
            out.to_pdb("file_should_not_exist.pdb")

        molecule.generate_conformers(n_conformers=1)
        out.positions = molecule.conformers[0]

        out.to_pdb("out.pdb")

        md.load("out.pdb")

        with pytest.raises(UnsupportedExportError):
            out.to_pdb("file_should_not_exist.pdb", writer="magik")
