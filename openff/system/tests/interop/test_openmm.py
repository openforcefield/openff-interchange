import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.tests.test_forcefield import create_ethanol
from openff.toolkit.tests.utils import get_data_file_path
from openff.toolkit.topology import Molecule, Topology
from simtk import openmm
from simtk import unit as simtk_unit
from simtk.openmm import app

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.drivers.openmm import _get_openmm_energies, get_openmm_energies
from openff.system.exceptions import (
    UnimplementedCutoffMethodError,
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.system.interop.openmm import from_openmm
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
        "result": openmm.NonbondedForce.LJPME,
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
        "result": UnimplementedCutoffMethodError,
    },
    {
        "vdw_method": "cutoff",
        "electrostatics_method": "reaction-field",
        "periodic": False,
        "result": UnimplementedCutoffMethodError,
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
        openmm_system = openff_system.to_openmm(combine_nonbonded_forces=True)
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
            openff_system.to_openmm(combine_nonbonded_forces=True)
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

    with pytest.raises(UnsupportedExportError, match="default NonbondedForce"):
        openff_sys.to_openmm(combine_nonbonded_forces=True)


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
    mol.conformers[0] -= np.min(mol.conformers) * simtk_unit.angstrom

    top.box_vectors = np.eye(3) * np.asarray([15, 15, 15]) * simtk_unit.nanometer

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 3 * simtk_unit.nanometer]
        )
        positions = positions * simtk_unit.angstrom

    toolkit_system = parsley.create_openmm_system(top)

    native_system = System.from_smirnoff(force_field=parsley, topology=top).to_openmm()

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


@pytest.mark.xfail(
    reason="from_openmm does not correctly import vdW parameters from custom forces."
)
@pytest.mark.slow
def test_openmm_roundtrip():
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    omm_top = top.to_openmm()

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    off_sys.box = [4, 4, 4]
    off_sys.positions = mol.conformers[0].value_in_unit(simtk_unit.nanometer)

    omm_sys = off_sys.to_openmm()

    converted = from_openmm(
        topology=omm_top,
        system=omm_sys,
    )

    converted.topology = off_sys.topology
    converted.box = off_sys.box
    converted.positions = off_sys.positions

    get_openmm_energies(off_sys).compare(
        get_openmm_energies(converted),
        custom_tolerances={"Nonbonded": 1.5 * simtk_unit.kilojoule_per_mole},
    )


@pytest.mark.slow
def test_combine_nonbonded_forces():

    mol = Molecule.from_smiles("ClC#CCl")
    mol.name = "HPER"
    mol.generate_conformers(n_conformers=1)

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    out = System.from_smirnoff(force_field=parsley, topology=mol.to_topology())
    out.box = [4, 4, 4]
    out.positions = mol.conformers[0]

    num_forces_combined = out.to_openmm(combine_nonbonded_forces=True).getNumForces()
    num_forces_uncombined = out.to_openmm(combine_nonbonded_forces=False).getNumForces()

    # The "new" forces are the split-off vdW forces, the 1-4 vdW, and the 1-4 electrostatics
    assert num_forces_combined + 3 == num_forces_uncombined

    get_openmm_energies(out, combine_nonbonded_forces=False).compare(
        get_openmm_energies(out, combine_nonbonded_forces=True),
    )
