import math

import numpy
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler
from openff.units import unit
from openff.utilities import get_data_file_path, has_package
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import (
    MoleculeWithConformer,
    _BaseTest,
    get_test_file_path,
)
from openff.interchange._tests.unit_tests.plugins.test_smirnoff_plugins import (
    TestDoubleExponential,
)
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.interchange.exceptions import (
    InterchangeException,
    MissingPositionsError,
    PluginCompatibilityError,
    UnsupportedExportError,
)
from openff.interchange.interop.openmm import (
    from_openmm,
    to_openmm_positions,
    to_openmm_topology,
)

if has_package("openmm"):
    import openmm
    import openmm.app
    import openmm.unit

    nonbonded_methods = [
        {
            "vdw_periodic": "cutoff",
            "vdw_nonperiodic": "no-cutoff",
            "electrostatics_periodic": "PME",
            "periodic": True,
            "result": openmm.NonbondedForce.PME,
        },
        {
            "vdw_periodic": "cutoff",
            "vdw_nonperiodic": "no-cutoff",
            "electrostatics_periodic": "PME",
            "periodic": False,
            "result": openmm.NonbondedForce.NoCutoff,
        },
    ]

else:
    nonbonded_methods = list()


def _get_num_virtual_sites(openmm_topology: "openmm.app.Topology") -> int:
    return sum(atom.element is None for atom in openmm_topology.atoms())


def _compare_openmm_topologies(
    top1: "openmm.app.Topology",
    top2: "openmm.app.Topology",
):
    """
    In lieu of first-class serializaiton in OpenMM (https://github.com/openmm/openmm/issues/1543),
    do some quick heuristics to roughly compare two OpenMM Topology objects.
    """
    for method_name in [
        "getNumAtoms",
        "getNumBonds",
        "getNumChains",
        "getNumResidues",
    ]:
        assert getattr(top1, method_name)() == getattr(top2, method_name)()

    assert (top1.getPeriodicBoxVectors() == top2.getPeriodicBoxVectors()).all()


@skip_if_missing("openmm")
class TestOpenMM(_BaseTest):
    @pytest.mark.parametrize("inputs", nonbonded_methods)
    def test_openmm_nonbonded_methods(self, inputs, sage, ethanol):
        """See test_nonbonded_method_resolution in openff.toolkit._tests/test_forcefield.py"""
        electrostatics_method = inputs["electrostatics_periodic"]
        periodic = inputs["periodic"]
        result = inputs["result"]

        molecules = [ethanol]

        pdbfile = openmm.app.PDBFile(
            get_data_file_path("systems/test_systems/1_ethanol.pdb", "openff.toolkit"),
        )
        topology = Topology.from_openmm(pdbfile.topology, unique_molecules=molecules)

        if not periodic:
            topology.box_vectors = None

        if inputs["periodic"]:
            sage.get_parameter_handler("vdW", {}).periodic_method = inputs[
                "vdw_periodic"
            ]
        else:
            sage.get_parameter_handler("vdW", {}).nonperiodic_method = inputs[
                "vdw_nonperiodic"
            ]

        sage.get_parameter_handler(
            "Electrostatics",
            {},
        ).periodic_potential = electrostatics_method
        interchange = Interchange.from_smirnoff(
            force_field=sage,
            topology=topology,
        )
        if type(result) is int:
            nonbonded_method = result
            # The method is validated and may raise an exception if it's not supported.
            if inputs["periodic"]:
                sage.get_parameter_handler("vdW", {}).periodic_method = inputs[
                    "vdw_periodic"
                ]
            else:
                sage.get_parameter_handler("vdW", {}).nonperiodic_method = inputs[
                    "vdw_nonperiodic"
                ]

            sage.get_parameter_handler(
                "Electrostatics",
                {},
            ).periodic_potential = electrostatics_method
            interchange = Interchange.from_smirnoff(
                force_field=sage,
                topology=topology,
            )
            openmm_system = interchange.to_openmm(combine_nonbonded_forces=True)
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    assert force.getNonbondedMethod() == nonbonded_method
                    break
            else:
                raise Exception
        elif issubclass(result, InterchangeException):
            exception = result
            with pytest.raises(exception):
                interchange.to_openmm(combine_nonbonded_forces=True)
        else:
            raise Exception

    def test_combine_nonbonded_forces_nondefault_mixing_rule(self, ethanol):
        forcefield = ForceField(
            get_data_file_path(
                "test_forcefields/test_forcefield.offxml",
                "openff.toolkit",
            ),
        )
        openff_sys = Interchange.from_smirnoff(
            force_field=forcefield,
            topology=ethanol.to_topology(),
        )

        openff_sys["vdW"].mixing_rule = "foo"

        with pytest.raises(UnsupportedExportError, match="only supports L.*foo"):
            openff_sys.to_openmm(combine_nonbonded_forces=True)

    def test_geometric_mixing_rule(self):
        forcefield = ForceField(
            get_data_file_path(
                "test_forcefields/test_forcefield.offxml",
                "openff.toolkit",
            ),
        )

        molecule = Molecule.from_mapped_smiles(
            "[H:5][C:2]([H:6])([C:3]([H:7])([H:8])[Br:4])[Cl:1]",
        )
        topology = Topology.from_molecules(molecule)
        topology.box_vectors = [30, 30, 30] * unit.angstrom

        interchange = Interchange.from_smirnoff(
            force_field=forcefield,
            topology=topology,
        )

        # The toolkit doesn't allow
        # >>> forcefield["vdW"].combining_rules = "Geometric"
        # so we have to do this instead set it after parameterization.
        # https://github.com/openforcefield/openff-toolkit/blob/0.11.4/openff/toolkit/typing/engines/smirnoff/parameters.py#L2844-L2846

        interchange["vdW"].mixing_rule = "geometric"

        system = interchange.to_openmm(combine_nonbonded_forces=False)

        for force in system.getForces():
            if isinstance(force, openmm.CustomNonbondedForce):
                vdw_force = force
                break
        else:
            raise RuntimeError("Could not find custom non-bonded force.")

        for force in system.getForces():
            if isinstance(force, openmm.CustomBondForce):
                if "epsilon" in force.getEnergyFunction():
                    vdw_14_force = force
                    break
        else:
            raise RuntimeError("Could not find 1-4 vdW force.")

        cl_parameters = vdw_force.getParticleParameters(0)
        br_parameters = vdw_force.getParticleParameters(3)

        assert cl_parameters[0] == forcefield["vdW"].get_parameter(
            {"smirks": "[#17:1]"},
        )[0].sigma.m_as(unit.nanometer)
        assert cl_parameters[1] == forcefield["vdW"].get_parameter(
            {"smirks": "[#17:1]"},
        )[0].epsilon.m_as(unit.kilojoule_per_mole)

        assert br_parameters[0] == forcefield["vdW"].get_parameter(
            {"smirks": "[#35:1]"},
        )[0].sigma.m_as(unit.nanometer)
        assert br_parameters[1] == forcefield["vdW"].get_parameter(
            {"smirks": "[#35:1]"},
        )[0].epsilon.m_as(unit.kilojoule_per_mole)

        for index in range(vdw_14_force.getNumBonds()):
            particle1, particle2, parameters = vdw_14_force.getBondParameters(index)

            if particle1 == 0 and particle2 == 3:
                expected_sigma = numpy.sqrt(cl_parameters[0] * br_parameters[0])
                expected_epsilon = forcefield["vdW"].scale14 * numpy.sqrt(
                    cl_parameters[1] * br_parameters[1],
                )

                assert parameters[0] == expected_sigma
                assert parameters[1] == expected_epsilon
                break

            else:
                raise Exception("Did not find 1-4 Cl-Br interaction.")

    @pytest.mark.xfail(reason="Broken because of splitting non-bonded forces")
    @pytest.mark.slow()
    @pytest.mark.parametrize("mol_smi", ["C", "CC", "CCO"])
    def test_openmm_roundtrip(self, sage, mol_smi):
        topology = MoleculeWithConformer.from_smiles(mol_smi).to_topology()

        interchange = Interchange.from_smirnoff(sage, topology)

        interchange.box = [4, 4, 4]

        converted = from_openmm(
            topology=interchange.to_openmm_topology(),
            system=interchange.to_openmm(combine_nonbonded_forces=True),
        )

        converted.box = interchange.box
        converted.positions = interchange.positions

        get_openmm_energies(interchange).compare(
            get_openmm_energies(converted, combine_nonbonded_forces=True),
        )

    @pytest.mark.xfail(reason="Broken because of splitting non-bonded forces")
    @pytest.mark.slow()
    def test_combine_nonbonded_forces(self, sage):
        topology = MoleculeWithConformer.from_smiles(
            "ClC#CCl",
            name="HPER",
        ).to_topology()

        out = Interchange.from_smirnoff(force_field=sage, topology=topology)
        out.box = [4, 4, 4]

        num_forces_combined = out.to_openmm(
            combine_nonbonded_forces=True,
        ).getNumForces()
        num_forces_uncombined = out.to_openmm(
            combine_nonbonded_forces=False,
        ).getNumForces()

        # The "new" forces are the split-off vdW forces, the 1-4 vdW, and the 1-4 electrostatics
        assert num_forces_combined + 3 == num_forces_uncombined

        separate = get_openmm_energies(out, combine_nonbonded_forces=False)
        combined = get_openmm_energies(out, combine_nonbonded_forces=True)

        assert (
            separate["vdW"] + separate["Electrostatics"] - combined["Nonbonded"]
        ).m < 0.001

    def test_openmm_no_angle_force_if_constrained(self, water, sage):
        topology = water.to_topology()
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        # Sage includes angle parameters for water and also TIP3P constraints
        interchange = Interchange.from_smirnoff(sage, topology)
        openmm_system = interchange.to_openmm(combine_nonbonded_forces=True)

        # The only angle in the system (H-O-H) includes bonds with constrained lengths
        # and a constrained angle, so by convention a force should NOT be added
        for force in openmm_system.getForces():
            if type(force) is openmm.HarmonicAngleForce:
                assert force.getNumAngles() == 0
                break
        else:
            raise Exception("No HarmonicAngleForce found")

    @pytest.mark.skip(reason="Rewrite as a plugin")
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

    def test_openmm_no_valence_forces_with_no_handler(self, sage, ethanol):
        original_system = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True,
        )
        assert original_system.getNumForces() == 4

        sage.deregister_parameter_handler("Constraints")
        sage.deregister_parameter_handler("Bonds")

        no_bonds = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True,
        )
        assert no_bonds.getNumForces() == 3

        sage.deregister_parameter_handler("Angles")

        no_angles = Interchange.from_smirnoff(sage, [ethanol]).to_openmm(
            combine_nonbonded_forces=True,
        )
        assert no_angles.getNumForces() == 2

    def test_openmm_only_electrostatics_no_vdw(self):
        force_field_only_charges = ForceField(get_test_file_path("no_vdw.offxml"))
        molecule = Molecule.from_smiles("[H][Cl]")

        system = Interchange.from_smirnoff(
            force_field_only_charges,
            [molecule],
        ).to_openmm(
            combine_nonbonded_forces=True,
        )

        assert system.getForce(0).getParticleParameters(0)[0]._value == 1.0
        assert system.getForce(0).getParticleParameters(1)[0]._value == -1.0

    def test_nonstandard_cutoffs_match(self, sage):
        """Test that multiple nonbonded forces use the same cutoff."""
        topology = Molecule.from_smiles("C").to_topology()
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        cutoff = unit.Quantity(1.555, unit.nanometer)

        sage["vdW"].cutoff = cutoff

        interchange = Interchange.from_smirnoff(
            force_field=sage,
            topology=topology,
        )

        system = interchange.to_openmm(combine_nonbonded_forces=False)

        # For now, just make sure all non-bonded forces use the vdW handler's cutoff
        for force in system.getForces():
            if type(force) in (openmm.NonbondedForce, openmm.CustomNonbondedForce):
                assert force.getCutoffDistance().value_in_unit(
                    openmm.unit.nanometer,
                ) == pytest.approx(cutoff.m_as(unit.nanometer))


@skip_if_missing("openmm")
class TestOpenMMSwitchingFunction(_BaseTest):
    def test_switching_function_applied(self, sage, basic_top):
        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True,
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert force.getUseSwitchingFunction()
                assert force.getSwitchingDistance().value_in_unit(
                    openmm.unit.angstrom,
                ) == pytest.approx(8), force.getSwitchingDistance()

        assert found_force, "NonbondedForce not found in system"

    def test_switching_function_not_applied(self, sage, basic_top):
        sage["vdW"].switch_width = 0.0 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True,
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert not force.getUseSwitchingFunction()
                assert force.getSwitchingDistance() == -1 * openmm.unit.nanometer

        assert found_force, "NonbondedForce not found in system"

    def test_switching_function_nonstandard(self, sage, basic_top):
        sage["vdW"].switch_width = 0.12345 * unit.angstrom

        out = Interchange.from_smirnoff(force_field=sage, topology=basic_top).to_openmm(
            combine_nonbonded_forces=True,
        )

        found_force = False
        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                found_force = True
                assert force.getUseSwitchingFunction()
                assert (
                    force.getSwitchingDistance() - (9 - 0.12345) * openmm.unit.angstrom
                ) < 1e-10 * openmm.unit.angstrom

        assert found_force, "NonbondedForce not found in system"


@skip_if_missing("openmm")
class TestOpenMMWithPlugins(TestDoubleExponential):
    def test_combine_compatibility(self, de_force_field):
        out = Interchange.from_smirnoff(
            force_field=de_force_field,
            topology=[Molecule.from_smiles("CO")],
        )

        with pytest.raises(
            PluginCompatibilityError,
            match="failed a compatibility check",
        ) as exception:
            out.to_openmm(combine_nonbonded_forces=True)

        assert isinstance(exception.value.__cause__, AssertionError)

    def test_nocutoff_when_nonperiodic(self, de_force_field):
        system = Interchange.from_smirnoff(
            de_force_field,
            MoleculeWithConformer.from_smiles("CCO").to_topology(),
        ).to_openmm(combine_nonbonded_forces=False)

        for force in system.getForces():
            if type(force) in (
                openmm.NonbondedForce,
                openmm.CustomNonbondedForce,
            ):
                assert force.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff

    def test_double_exponential_create_simulation(self, de_force_field):
        from openff.toolkit.utils.openeye_wrapper import OpenEyeToolkitWrapper

        topology = MoleculeWithConformer.from_smiles("CCO").to_topology()
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        out = Interchange.from_smirnoff(
            de_force_field,
            topology,
        )

        system = out.to_openmm(combine_nonbonded_forces=False)

        simulation = openmm.app.Simulation(
            to_openmm_topology(out),
            system,
            openmm.LangevinIntegrator(300, 1, 0.002),
            openmm.Platform.getPlatformByName("CPU"),
        )

        simulation.context.setPositions(
            to_openmm_positions(out, include_virtual_sites=False),
        )
        simulation.context.setPeriodicBoxVectors(*out.box.to_openmm())

        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().in_units_of(openmm.unit.kilojoule_per_mole)

        if OpenEyeToolkitWrapper.is_available():
            expected_energy = 13.591709748611304
        else:
            expected_energy = 37.9516622967221

        # Different operating systems report different energies around 0.001 kJ/mol,
        # locally testing this should enable something like 1e-6 kJ/mol
        assert abs(energy._value - expected_energy) < 3e-3


@skip_if_missing("openmm")
@pytest.mark.slow()
class TestOpenMMVirtualSites:
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

    def test_valence_term_paticle_index_offsets(self, water, tip5p):
        out = Interchange.from_smirnoff(tip5p, [water, water]).to_openmm(
            combine_nonbonded_forces=True,
        )

        # NonbondedForce and HarmonicBondForce; no HarmonicAngleForce (even if there were force
        # field parameters added, the current implementation would not add an angle force because
        # H-O-H is fully constrained)
        assert out.getNumForces() == 2

        # Particle indexing is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        #                      O, H, H, O, H, H, VS, VS, VS, VS
        for index in range(6):
            assert out.getParticleMass(index)._value > 0.0

        for index in range(6, 10):
            assert out.getParticleMass(index)._value == 0.0
            assert isinstance(out.getVirtualSite(index), openmm.VirtualSite)


@skip_if_missing("openmm")
class TestOpenMMVirtualSiteExclusions(_BaseTest):
    def test_tip5p_num_exceptions(self, water):
        tip5p = ForceField(get_test_file_path("tip5p.offxml"))

        out = Interchange.from_smirnoff(tip5p, [water]).to_openmm(
            combine_nonbonded_forces=True,
        )

        # In a TIP5P water    expected exceptions include (total 10)
        #
        # V(3)  V(4)          Oxygen to hydrogens and particles (4)
        #    \ /                - (0, 1), (0, 2), (0, 3), (0, 4)
        #     O(0)            Hyrogens to virtual particles (4)
        #    / \                - (1, 3), (1, 4), (2, 3), (2, 4)
        # H(1)  H(2)          Hydrogens and virtual particles to each other (2)
        #                       - (1, 2), (3, 4)

        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getNumExceptions() == 10

    def test_dichloroethane_exceptions(self, sage):
        """Test a case in which a parent's 1-4 exceptions must be 'imported'."""
        from openff.toolkit._tests.mocking import VirtualSiteMocking

        # This molecule has heavy atoms with indices (1-indexed) CL1, C2, C3, Cl4,
        # resulting in 1-4 interactions between the Cl-Cl pair and some Cl-H pairs
        dichloroethane = Molecule.from_mapped_smiles(
            "[Cl:1][C:2]([H:5])([H:6])[C:3]([H:7])([H:8])[Cl:4]",
        )

        # This parameter pulls 0.1 and 0.2e from Cl (parent) and C, respectively, and has
        # LJ parameters of 4 A, 3 kJ/mol
        parameter = VirtualSiteMocking.bond_charge_parameter("[Cl:1]-[C:2]")

        handler = VirtualSiteHandler(version="0.3")
        handler.add_parameter(parameter=parameter)

        sage.register_parameter_handler(handler)

        system = Interchange.from_smirnoff(sage, [dichloroethane]).to_openmm(
            combine_nonbonded_forces=True,
        )

        assert system.isVirtualSite(8)
        assert system.isVirtualSite(9)

        non_bonded_force = [
            f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
        ][0]

        for exception_index in range(non_bonded_force.getNumExceptions()):
            p1, p2, q, sigma, epsilon = non_bonded_force.getExceptionParameters(
                exception_index,
            )
            if p2 == 8:
                # Parent Cl, adjacent C and its bonded H, and the 1-3 C
                if p1 in (0, 1, 2, 4, 5):
                    assert q._value == epsilon._value == 0.0
                # 1-4 Cl or 1-4 Hs
                if p1 in (3, 6, 7):
                    for value in (q, sigma, epsilon):
                        assert value._value != 0, (q, sigma, epsilon)
            if p2 == 9:
                if p1 in (3, 1, 2, 6, 7):
                    assert q._value == epsilon._value == 0.0
                if p1 in (0, 4, 5):
                    for value in (q, sigma, epsilon):
                        assert value._value != 0, (q, sigma, epsilon)


@skip_if_missing("openmm")
class TestToOpenMMTopology(_BaseTest):
    def test_num_virtual_sites(self, water, tip4p):
        out = Interchange.from_smirnoff(tip4p, [water])

        assert _get_num_virtual_sites(to_openmm_topology(out)) == 1

        # TODO: Monkeypatch Topology.to_openmm() and emit a warning when it seems
        #       to be used while virtual sites are present in a handler
        assert _get_num_virtual_sites(out.topology.to_openmm()) == 0

    def test_interchange_method(self, water, tip4p):
        """
        Ensure similar-ish behavior between `to_openmm_topology` as a standalone function
        and as the wrapped method of the same name on the `Interchange` class.
        """
        topology = water.to_topology()
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        out = Interchange.from_smirnoff(tip4p, topology)

        _compare_openmm_topologies(out.to_openmm_topology(), to_openmm_topology(out))

    @pytest.mark.parametrize("ensure_unique_atom_names", [True, "residues", "chains"])
    def test_assign_unique_atom_names(self, ensure_unique_atom_names, sage):
        """
        Ensure that OFF topologies with no pre-existing atom names have unique
        atom names applied when being converted to openmm
        """
        # Create OpenFF topology with 1 ethanol and 2 benzenes.
        ethanol = Molecule.from_smiles("CCO")
        benzene = Molecule.from_smiles("c1ccccc1")
        off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])

        # This test uses molecules with no hierarchy schemes, so the parametrized
        # ensure_unique_atom_names values should behave identically.
        assert not any(
            [mol._hierarchy_schemes for mol in off_topology.molecules],
        ), "Test assumes no hierarchy schemes"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        omm_topology = interchange.to_openmm_topology(
            ensure_unique_atom_names=ensure_unique_atom_names,
        )
        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)
        # There should be 6 unique Cs, 6 unique Hs, and 1 unique O, for a total of 13 unique atom names
        assert len(atom_names) == 13

    @pytest.mark.parametrize("ensure_unique_atom_names", [True, "residues", "chains"])
    def test_assign_some_unique_atom_names(self, ensure_unique_atom_names, sage):
        """
        Ensure that OFF topologies with some pre-existing atom names have unique
        atom names applied to the other atoms when being converted to openmm
        """
        # Create OpenFF topology with 1 ethanol and 2 benzenes.
        ethanol = Molecule.from_smiles("CCO")
        for atom in ethanol.atoms:
            atom.name = f"AT{atom.molecule_atom_index}"
        benzene = Molecule.from_smiles("c1ccccc1")
        off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])

        # This test uses molecules with no hierarchy schemes, so the parametrized
        # ensure_unique_atom_names values should behave identically.
        assert not any(
            [mol._hierarchy_schemes for mol in off_topology.molecules],
        ), "Test assumes no hierarchy schemes"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        omm_topology = interchange.to_openmm_topology(
            ensure_unique_atom_names=ensure_unique_atom_names,
        )
        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)
        # There should be 9 "ATOM#"-labeled atoms, 6 unique Cs, and 6 unique Hs,
        # for a total of 21 unique atom names
        assert len(atom_names) == 21

    @pytest.mark.parametrize("ensure_unique_atom_names", [True, "residues", "chains"])
    def test_assign_unique_atom_names_some_duplicates(
        self,
        ensure_unique_atom_names,
        sage,
    ):
        """
        Ensure that OFF topologies where some molecules have invalid/duplicate
        atom names have unique atom names applied while the other molecules are unaffected.
        """
        # Create OpenFF topology with 1 ethanol and 2 benzenes.
        ethanol = Molecule.from_smiles("CCO")

        # Assign duplicate atom names in ethanol (two AT0s)
        ethanol_atom_names_with_duplicates = [f"AT{i}" for i in range(ethanol.n_atoms)]
        ethanol_atom_names_with_duplicates[1] = "AT0"
        for atom, atom_name in zip(ethanol.atoms, ethanol_atom_names_with_duplicates):
            atom.name = atom_name

        # Assign unique atom names in benzene
        benzene = Molecule.from_smiles("c1ccccc1")
        benzene_atom_names = [f"AT{i}" for i in range(benzene.n_atoms)]
        for atom, atom_name in zip(benzene.atoms, benzene_atom_names):
            atom.name = atom_name

        off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])

        # This test uses molecules with no hierarchy schemes, so the parametrized
        # ensure_unique_atom_names values should behave identically.
        assert not any(
            [mol._hierarchy_schemes for mol in off_topology.molecules],
        ), "Test assumes no hierarchy schemes"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        omm_topology = interchange.to_openmm_topology(
            ensure_unique_atom_names=ensure_unique_atom_names,
        )
        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)

        # There should be  12 "AT#"-labeled atoms (from benzene), 2 unique Cs,
        # 1 unique O, and 6 unique Hs, for a total of 21 unique atom names
        assert len(atom_names) == 21

    def test_do_not_assign_unique_atom_names(self, sage):
        """
        Test disabling unique atom name assignment in Topology.to_openmm
        """
        # Create OpenFF topology with 1 ethanol and 2 benzenes.
        ethanol = Molecule.from_smiles("CCO")
        for atom in ethanol.atoms:
            atom.name = "eth_test"

        benzene = Molecule.from_smiles("c1ccccc1")
        benzene.atoms[0].name = "bzn_test"

        off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])

        interchange = Interchange.from_smirnoff(sage, off_topology)

        omm_topology = interchange.to_openmm_topology(ensure_unique_atom_names=False)
        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)
        # There should be 9 atom named "eth_test", 1 atom named "bzn_test",
        # and 12 atoms named "", for a total of 3 unique atom names
        assert len(atom_names) == 3

    @pytest.mark.slow()
    @pytest.mark.parametrize("explicit_arg", [True, False])
    def test_preserve_per_residue_unique_atom_names(self, explicit_arg, sage):
        """
        Test that to_openmm preserves atom names that are unique per-residue by default
        """
        # Create a topology from a capped dialanine
        peptide = Molecule.from_polymer_pdb(
            get_data_file_path(
                "proteins/MainChain_ALA_ALA.pdb",
                "openff.toolkit",
            ),
        )
        off_topology = Topology.from_molecules([peptide])

        # Assert the test's assumptions
        _ace, ala1, ala2, _nme = off_topology.hierarchy_iterator("residues")
        assert [a.name for a in ala1.atoms] == [
            a.name for a in ala2.atoms
        ], "Test assumes both alanines have same atom names"

        for res in off_topology.hierarchy_iterator("residues"):
            res_atomnames = [atom.name for atom in res.atoms]
            assert len(set(res_atomnames)) == len(
                res_atomnames,
            ), f"Test assumes atom names are already unique per-residue in {res}"

        # Record the initial atom names
        init_atomnames = [str(atom.name) for atom in off_topology.atoms]

        interchange = Interchange.from_smirnoff(sage, off_topology)

        # Perform the test
        if explicit_arg:
            omm_topology = interchange.to_openmm_topology(
                ensure_unique_atom_names="residues",
            )
        else:
            omm_topology = interchange.to_openmm_topology()

        # Check that the atom names were preserved
        final_atomnames = [str(atom.name) for atom in omm_topology.atoms()]
        assert final_atomnames == init_atomnames

    @pytest.mark.slow()
    @pytest.mark.parametrize("explicit_arg", [True, False])
    def test_generate_per_residue_unique_atom_names(self, explicit_arg, sage):
        """
        Test that to_openmm generates atom names that are unique per-residue
        """
        # Create a topology from a capped dialanine
        peptide = Molecule.from_polymer_pdb(
            get_data_file_path("proteins/MainChain_ALA_ALA.pdb", "openff.toolkit"),
        )
        off_topology = Topology.from_molecules([peptide])

        # Remove atom names from some residues, make others have duplicate atom names
        ace, ala1, ala2, nme = off_topology.hierarchy_iterator("residues")
        for atom in ace.atoms:
            atom._name = None
        for atom in ala1.atoms:
            atom.name = ""
        for atom in ala2.atoms:
            atom.name = "ATX2"
        for atom in nme.atoms:
            if atom.name == "H2":
                atom.name = "H1"
                break

        # Assert assumptions
        for res in off_topology.hierarchy_iterator("residues"):
            res_atomnames = [atom.name for atom in res.atoms]
            assert len(set(res_atomnames)) != len(
                res_atomnames,
            ), f"Test assumes atom names are not unique per-residue in {res}"
        assert off_topology.n_atoms == 32, "Test assumes topology has 32 atoms"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        # Perform the test
        if explicit_arg:
            omm_topology = interchange.to_openmm_topology(
                ensure_unique_atom_names="residues",
            )
        else:
            omm_topology = interchange.to_openmm_topology()

        # Check that the atom names are now unique per-residue but not per-molecule
        for res in omm_topology.residues():
            res_atomnames = [atom.name for atom in res.atoms()]
            assert len(set(res_atomnames)) == len(
                res_atomnames,
            ), f"Final atom names are not unique in residue {res}"

        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)
        assert (
            len(atom_names) < 32
        ), "There should be duplicate atom names in this output topology"

    @pytest.mark.parametrize("ensure_unique_atom_names", ["chains", True])
    def test_generate_per_molecule_unique_atom_names_with_residues(
        self,
        ensure_unique_atom_names,
        sage,
    ):
        """
        Test that to_openmm can generate atom names that are unique per-molecule
        when the topology has residues
        """
        # Create a topology from a capped dialanine
        peptide = Molecule.from_polymer_pdb(
            get_data_file_path("proteins/MainChain_ALA_ALA.pdb", "openff.toolkit"),
        )
        off_topology = Topology.from_molecules([peptide])

        # Remove atom names from some residues, make others have duplicate atom names
        ace, ala1, ala2, nme = off_topology.hierarchy_iterator("residues")
        for atom in ace.atoms:
            atom._name = None
        for atom in ala1.atoms:
            atom.name = ""
        for atom in ala2.atoms:
            atom.name = "ATX2"
        for atom in nme.atoms:
            if atom.name == "H2":
                atom.name = "H1"
                break

        # Assert assumptions
        for res in off_topology.hierarchy_iterator("residues"):
            res_atomnames = [atom.name for atom in res.atoms]
            assert len(set(res_atomnames)) != len(
                res_atomnames,
            ), f"Test assumes atom names are not unique per-residue in {res}"
        assert off_topology.n_atoms == 32, "Test assumes topology has 32 atoms"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        # Perform the test
        omm_topology = interchange.to_openmm_topology(
            ensure_unique_atom_names=ensure_unique_atom_names,
        )

        # Check that the atom names are now unique across the topology (of 1 molecule)
        atom_names = set()
        for atom in omm_topology.atoms():
            atom_names.add(atom.name)
        assert (
            len(atom_names) == 32
        ), "There should not be duplicate atom names in this output topology"

    @pytest.mark.parametrize(
        "ensure_unique_atom_names",
        [True, "residues", "chains", False],
    )
    def test_to_openmm_copies_molecules(self, ensure_unique_atom_names, sage):
        """
        Check that generating new atom names doesn't affect the input topology
        """
        # Create OpenFF topology with 1 ethanol and 2 benzenes.
        ethanol = Molecule.from_smiles("CCO")
        for atom in ethanol.atoms:
            atom.name = f"AT{atom.molecule_atom_index}"
        benzene = Molecule.from_smiles("c1ccccc1")
        off_topology = Topology.from_molecules(molecules=[ethanol, benzene, benzene])

        # This test uses molecules with no hierarchy schemes, so the parametrized
        # ensure_unique_atom_names values should behave identically (except False).
        assert not any(
            [mol._hierarchy_schemes for mol in off_topology.molecules],
        ), "Test assumes no hierarchy schemes"

        interchange = Interchange.from_smirnoff(sage, off_topology)

        # Record the initial atom names to compare to later
        init_atomnames = [str(atom.name) for atom in interchange.topology.atoms]

        omm_topology = interchange.to_openmm_topology(
            ensure_unique_atom_names=ensure_unique_atom_names,
        )

        # Get the atom names back from the initial molecules after calling to_openmm
        final_atomnames_mols = [
            atom.name for atom in [*ethanol.atoms, *benzene.atoms, *benzene.atoms]
        ]
        # Get the atom names back from the initial topology after calling to_openmm
        final_atomnames_offtop = [atom.name for atom in off_topology.atoms]
        # Get the atom names back from the new OpenMM topology
        final_atomnames_ommtop = [atom.name for atom in omm_topology.atoms()]

        # Check the appropriate properties!
        assert (
            init_atomnames == final_atomnames_mols
        ), "Molecules' atom names were changed"
        assert (
            init_atomnames == final_atomnames_offtop
        ), "Topology's atom names were changed"
        if ensure_unique_atom_names:
            assert (
                init_atomnames != final_atomnames_ommtop
            ), "New atom names should've been generated but weren't"


@skip_if_missing("openmm")
class TestToOpenMMPositions(_BaseTest):
    def test_missing_positions(self):
        with pytest.raises(
            MissingPositionsError,
            match=r"are required.*\.positions=None",
        ):
            to_openmm_positions(Interchange())

    @pytest.mark.parametrize("include_virtual_sites", [True, False])
    def test_positions_basic(self, include_virtual_sites, water, tip4p):
        out = Interchange.from_smirnoff(tip4p, [water])

        positions = to_openmm_positions(
            out,
            include_virtual_sites=include_virtual_sites,
        )

        assert isinstance(positions, openmm.unit.Quantity)
        assert positions.shape == (4, 3) if include_virtual_sites else (3, 3)

        numpy.testing.assert_allclose(
            positions.value_in_unit(openmm.unit.angstrom)[:3],
            water.conformers[0].m_as(unit.angstrom),
        )

    @pytest.mark.parametrize("include_virtual_sites", [True, False])
    def test_given_positions(self, include_virtual_sites, water, tip4p):
        """Test issue #616"""
        import openmm.unit

        topology = Topology.from_molecules([water, water])
        out = Interchange.from_smirnoff(tip4p, topology)

        # Approximate conformer position with a duplicate 5 A away in x
        out.positions = unit.Quantity(
            numpy.array(
                [
                    [0.85, 1.17, 0.84],
                    [1.51, 0.47, 0.75],
                    [0.0, 0.71, 0.76],
                    [5.85, 1.17, 0.84],
                    [6.51, 0.47, 0.75],
                    [5.0, 0.71, 0.76],
                ],
            ),
            unit.angstrom,
        )

        positions = to_openmm_positions(
            out,
            include_virtual_sites=include_virtual_sites,
        )

        assert isinstance(positions, openmm.unit.Quantity)
        assert str(positions.unit) == "nanometer"

        assert numpy.allclose(
            (positions[3:6, :] - positions[:3, :]).value_in_unit(openmm.unit.angstrom),
            [5, 0, 0],
        )

        if include_virtual_sites:
            assert numpy.allclose(
                (positions[-1, :] - positions[-2, :]).value_in_unit(
                    openmm.unit.angstrom,
                ),
                [5, 0, 0],
            )


@skip_if_missing("mdtraj")
@skip_if_missing("openmm")
class TestOpenMMToPDB(_BaseTest):
    def test_to_pdb(self, sage, water):
        import mdtraj

        out = Interchange.from_smirnoff(sage, water.to_topology())
        out.to_pdb("out.pdb")

        mdtraj.load("out.pdb")

        out.positions = None

        with pytest.raises(MissingPositionsError):
            out.to_pdb("file_should_not_exist.pdb")


@skip_if_missing("openmm")
class TestBuckingham:
    def test_water_with_virtual_sites(self, water):
        force_field = ForceField(
            get_test_file_path("buckingham_virtual_sites.offxml"),
            load_plugins=True,
        )

        interchange = Interchange.from_smirnoff(
            force_field=force_field,
            topology=water.to_topology(),
            box=[4, 4, 4],
        )

        with pytest.raises(PluginCompatibilityError):
            interchange.to_openmm(combine_nonbonded_forces=True)

        system = interchange.to_openmm(combine_nonbonded_forces=False)

        assert system.getNumForces() == 4

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                electrostatics = force
                continue
            elif isinstance(force, openmm.CustomNonbondedForce):
                vdw = force
                continue
            elif isinstance(force, openmm.CustomBondForce):
                if "qq" in force.getEnergyFunction():
                    electrostatics14 = force
                    continue

        assert system.getNumParticles() == 4

        masses = [15.99943, 1.007947, 1.007947, 0.0]

        for particle_index in range(system.getNumParticles()):
            assert system.isVirtualSite(particle_index) == (particle_index == 3)
            assert system.getParticleMass(particle_index)._value == pytest.approx(
                masses[particle_index],
            )

        charges = openmm.unit.Quantity(
            [0.0, 0.53254, 0.53254, -1.06508],
            openmm.unit.elementary_charge,
        )

        for index, charge in enumerate(charges):
            assert electrostatics.getParticleParameters(index)[0] == charge

        for index in range(vdw.getNumParticles()):
            parameters = vdw.getParticleParameters(index)
            for p in parameters:
                assert (p == 0) == (index > 0)

        assert vdw.getParticleParameters(0) == (1600000.0, 42.0, 0.003)

        # This test should be replaced with one that uses a more complex
        # system than a single water molecule and look at vdw14 force
        assert electrostatics14.getNumBonds() == 0

        with pytest.raises(PluginCompatibilityError):
            get_openmm_energies(interchange, combine_nonbonded_forces=True)

        with pytest.warns(
            UserWarning,
            match="energies from split forces with virtual sites",
        ):
            assert not math.isnan(
                get_openmm_energies(
                    interchange,
                    combine_nonbonded_forces=False,
                ).total_energy.m,
            )


@skip_if_missing("openmm")
class TestGBSA(_BaseTest):
    def test_create_gbsa(self, gbsa_force_field):
        interchange = Interchange.from_smirnoff(
            force_field=gbsa_force_field,
            topology=MoleculeWithConformer.from_smiles("CCO").to_topology(),
            box=[4, 4, 4] * unit.nanometer,
        )

        assert get_openmm_energies(interchange).total_energy is not None

    def test_cannot_split_nonbonded_forces(self, gbsa_force_field):
        with pytest.raises(UnsupportedExportError, match="exactly one"):
            Interchange.from_smirnoff(
                force_field=gbsa_force_field,
                topology=MoleculeWithConformer.from_smiles("CCO").to_topology(),
                box=[4, 4, 4] * unit.nanometer,
            ).to_openmm(combine_nonbonded_forces=False)

    def test_no_cutoff(self, gbsa_force_field):
        system = Interchange.from_smirnoff(
            force_field=gbsa_force_field,
            topology=MoleculeWithConformer.from_smiles("CCO").to_topology(),
            box=None,
        ).to_openmm(combine_nonbonded_forces=True)

        for force in system.getForces():
            if isinstance(force, openmm.CustomGBForce):
                assert force.getNonbondedMethod() == openmm.CustomGBForce.NoCutoff
                # This should be set to OpenMM's default, though not used
                assert force.getCutoffDistance() == 1.0 * openmm.unit.nanometer
                break

        else:
            raise Exception
