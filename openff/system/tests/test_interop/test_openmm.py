import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.tests.test_forcefield import create_ethanol
from openff.toolkit.tests.utils import compare_system_energies, get_data_file_path
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import SMIRNOFFSpecError
from simtk import openmm, unit
from simtk.openmm import app

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.exceptions import (
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.system.stubs import ForceField
from openff.system.utils import get_test_file_path

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
        "result": UnsupportedCutoffMethodError,
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "PME",
        "periodic": False,
        "result": UnsupportedCutoffMethodError,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "reaction-field",
        "periodic": True,
        "result": SMIRNOFFSpecError,  # UnimplementedCutoffMethodError,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "reaction-field",
        "periodic": False,
        "result": SMIRNOFFSpecError,  # UnimplementedCutoffMethodError,
    },
]
"""\
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "PME",
        "has_periodic_box": False,
        "omm_force": openmm.NonbondedForce.NoCutoff,
        "exception": None,
        "exception_match": "",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "Coulomb",
        "has_periodic_box": True,
        "omm_force": None,
        "exception": IncompatibleParameterError,
        "exception_match": "",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "Coulomb",
        "has_periodic_box": False,
        "omm_force": openmm.NonbondedForce.NoCutoff,
        "exception": None,
        "exception_match": "",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "reaction-field",
        "has_periodic_box": True,
        "omm_force": None,
        "exception": IncompatibleParameterError,
        "exception_match": "reaction-field",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "reaction-field",
        "has_periodic_box": False,
        "omm_force": None,
        "exception": SMIRNOFFSpecError,
        "exception_match": "reaction-field",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "PME",
        "has_periodic_box": True,
        "omm_force": openmm.NonbondedForce.LJPME,
        "exception": None,
        "exception_match": "",
    },
    {
        "vdw_method": "PME",
        "electrostatics_method": "PME",
        "has_periodic_box": False,
        "omm_force": openmm.NonbondedForce.NoCutoff,
        "exception": None,
        "exception_match": "",
    },
]
"""


@pytest.mark.parametrize("inputs", nonbonded_resolution_matrix)
def test_openmm_nonbonded_methods(inputs):
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
        openff_system = System.from_smirnoff(force_field=forcefield, topology=topology)
        openmm_system = openff_system.to_openmm()
        for force in openmm_system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getNonbondedMethod() == nonbonded_method
                break
        else:
            raise Exception
    elif issubclass(result, (BaseException, Exception)):
        exception = result
        with pytest.raises(exception):
            forcefield.get_parameter_handler("vdW", {}).method = vdw_method
            forcefield.get_parameter_handler(
                "Electrostatics", {}
            ).method = electrostatics_method
            openff_system = System.from_smirnoff(
                force_field=forcefield, topology=topology
            )
            openff_system.to_openmm()
    else:
        raise Exception("uh oh")


def test_unsupported_mixing_rule():
    molecules = [create_ethanol()]
    pdbfile = app.PDBFile(get_data_file_path("systems/test_systems/1_ethanol.pdb"))
    topology = OFFBioTop.from_openmm(pdbfile.topology, unique_molecules=molecules)
    topology.mdtop = md.Topology.from_openmm(topology.to_openmm())

    forcefield = ForceField("test_forcefields/test_forcefield.offxml")
    openff_sys = System.from_smirnoff(force_field=forcefield, topology=topology)

    openff_sys["vdW"].mixing_rule = "geometric"

    with pytest.raises(UnsupportedExportError, match="rule `geometric` not compat"):
        openff_sys.to_openmm()


@pytest.mark.slow
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
def test_from_openmm_single_mols(mol, n_mols):
    """
    Test that ForceField.create_openmm_system and System.to_openmm produce
    objects with similar energies

    TODO: Tighten tolerances
    TODO: Test periodic and non-periodic
    """

    parsley = ForceField(get_test_file_path("parsley.offxml"))

    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(n_mols * [mol])
    mol.conformers[0] -= np.min(mol.conformers) * unit.angstrom

    top.box_vectors = np.eye(3) * np.asarray([15, 15, 15]) * unit.nanometer

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 3 * unit.nanometer]
        )
        positions = positions * unit.angstrom

    toolkit_system = parsley.create_openmm_system(top)

    native_system = System.from_smirnoff(force_field=parsley, topology=top).to_openmm()

    compare_system_energies(
        system1=toolkit_system,
        system2=native_system,
        positions=positions,
        box_vectors=top.box_vectors,
        atol=1e-5,
    )
