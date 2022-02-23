import numpy as np
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
from openff.interchange.interop.openmm import (
    from_openmm,
    get_partial_charges_from_openmm_system,
)
from openff.interchange.tests import _BaseTest

nonbonded_resolution_matrix = [
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "PME",
        "periodic": True,
        "result": openmm.NonbondedForce.PME,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "PME",
        "periodic": False,
        "result": UnsupportedCutoffMethodError,
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "PME",
        "periodic": True,
        "result": openmm.NonbondedForce.LJPME,
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "PME",
        "periodic": False,
        "result": UnsupportedCutoffMethodError,
    },
]
# Revisit after OpenFF Toolkit >0.9.2 release
"""
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "reaction-field",
        "periodic": True,
        "result": UnimplementedCutoffMethodError,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "reaction-field",
        "periodic": False,
        "result": UnimplementedCutoffMethodError,
    },
"""


class TestOpenMM(_BaseTest):
    @pytest.mark.parametrize("inputs", nonbonded_resolution_matrix)
    def test_openmm_nonbonded_methods(self, inputs):
        """See test_nonbonded_method_resolution in openff/toolkit/tests/test_forcefield.py"""
        vdw_method = inputs["vdw_method"]
        electrostatics_method = inputs["electrostatics_method"]
        periodic = inputs["periodic"]
        result = inputs["result"]

        molecules = [create_ethanol()]
        forcefield = ForceField("test_forcefields/test_forcefield.offxml")

        pdbfile = app.PDBFile(get_data_file_path("systems/test_systems/1_ethanol.pdb"))
        topology = Topology.from_openmm(pdbfile.topology, unique_molecules=molecules)

        if not periodic:
            topology.box_vectors = None

        if type(result) == int:
            nonbonded_method = result
            # The method is validated and may raise an exception if it's not supported.
            forcefield.get_parameter_handler("vdW", {}).method = vdw_method
            forcefield.get_parameter_handler(
                "Electrostatics", {}
            ).method = electrostatics_method
            openff_interchange = Interchange.from_smirnoff(
                force_field=forcefield, topology=topology
            )
            openmm_system = openff_interchange.to_openmm(combine_nonbonded_forces=True)
            for force in openmm_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    assert force.getNonbondedMethod() == nonbonded_method
                    break
            else:
                raise Exception
        elif issubclass(result, (BaseException, Exception)):
            exception = result
            forcefield.get_parameter_handler("vdW", {}).method = vdw_method
            forcefield.get_parameter_handler(
                "Electrostatics", {}
            ).method = electrostatics_method
            openff_interchange = Interchange.from_smirnoff(
                force_field=forcefield, topology=topology
            )
            with pytest.raises(exception):
                openff_interchange.to_openmm(combine_nonbonded_forces=True)
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
    def test_from_openmm_single_mols(parsley, mol, n_mols):
        """
        Test that ForceField.create_openmm_system and Interchange.to_openmm produce
        objects with similar energies

        TODO: Tighten tolerances
        TODO: Test periodic and non-periodic
        """
        mol = Molecule.from_smiles(mol)
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules(n_mols * [mol])
        mol.conformers[0] -= np.min(mol.conformers)

        top.box_vectors = 15 * np.eye(3) * unit.nanometer

        if n_mols == 1:
            positions = mol.conformers[0]
        elif n_mols == 2:
            positions = np.concatenate(
                [mol.conformers[0], mol.conformers[0] + 3 * mol.conformers[0].units]
            )

        toolkit_system = parsley.create_openmm_system(top)

        native_system = Interchange.from_smirnoff(
            force_field=parsley, topology=top
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
    def test_openmm_roundtrip(self, parsley, mol_smi):
        mol = Molecule.from_smiles(mol_smi)
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()
        omm_top = top.to_openmm()

        off_sys = Interchange.from_smirnoff(parsley, top)

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
    def test_combine_nonbonded_forces(self, parsley):

        mol = Molecule.from_smiles("ClC#CCl")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(force_field=parsley, topology=mol.to_topology())
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

    def test_openmm_partial_charges(self, argon_ff, argon_top):
        omm_system = argon_ff.create_openmm_system(argon_top)
        partial_charges = get_partial_charges_from_openmm_system(omm_system)

        # assert isinstance(partial_charges, unit.Quantity)
        # assert partial_charges.units == unit.elementary_charge
        assert isinstance(partial_charges, list)
        assert np.allclose(partial_charges, np.zeros(4))  # .magnitude

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


@pytest.mark.slow()
class TestOpenMMVirtualSites(_BaseTest):
    @pytest.fixture()
    def parsley_with_sigma_hole(self, parsley):
        """Fixture that loads an SMIRNOFF XML for argon"""
        # TODO: Move this into BaseTest to that GROMACS and others can access it
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        sigma_type = VirtualSiteHandler.VirtualSiteBondChargeType(
            name="EP",
            smirks="[#6:1]-[#17:2]",
            distance=1.4 * unit.angstrom,
            type="BondCharge",
            match="once",
            charge_increment1=0.1 * unit.elementary_charge,
            charge_increment2=0.2 * unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=sigma_type)
        parsley.register_parameter_handler(virtual_site_handler)

        return parsley

    @pytest.fixture()
    def parsley_with_monovalent_lone_pair(self, parsley):
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
        parsley.register_parameter_handler(virtual_site_handler)

        return parsley

    def test_sigma_hole_example(self, parsley_with_sigma_hole):
        """Test that a single-molecule sigma hole example runs"""
        mol = Molecule.from_smiles("CCl")
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=parsley_with_sigma_hole, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_openmm_energies(out, combine_nonbonded_forces=True)

        compare_system_parameters(
            out.to_openmm(combine_nonbonded_forces=True),
            parsley_with_sigma_hole.create_openmm_system(mol.to_topology()),
        )
        """
        import numpy as np
        import parmed as pmd

        out.to_top("sigma.top")
        gmx_top = pmd.load_file("sigma.top")

        assert abs(np.sum([p.charge for p in gmx_top.atoms])) < 1e-3
        """

    def test_carbonyl_example(self, parsley_with_monovalent_lone_pair):
        """Test that a single-molecule DivalentLonePair example runs"""
        mol = Molecule.from_smiles("CC=O")
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=parsley_with_monovalent_lone_pair, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_openmm_energies(out, combine_nonbonded_forces=True)

        compare_system_parameters(
            out.to_openmm(combine_nonbonded_forces=True),
            parsley_with_monovalent_lone_pair.create_openmm_system(mol.to_topology()),
        )


class TestOpenMMToPDB(_BaseTest):
    def test_to_pdb(self, parsley):
        import mdtraj as md

        molecule = Molecule.from_smiles("O")
        molecule.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(parsley, molecule.to_topology())

        with pytest.raises(MissingPositionsError):
            out.to_pdb("file_should_not_exist.pdb")

        out.positions = molecule.conformers[0]

        out.to_pdb("out.pdb")

        md.load("out.pdb")

        with pytest.raises(UnsupportedExportError):
            out.to_pdb("file_should_not_exist.pdb", writer="magik")
