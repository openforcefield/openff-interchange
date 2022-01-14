from copy import deepcopy

import numpy as np
import openmm
import pytest
from openff.toolkit.tests.utils import requires_openeye
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    LibraryChargeHandler,
    UnassignedAngleParameterException,
    UnassignedBondParameterException,
    UnassignedProperTorsionParameterException,
)
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from openmm import unit as openmm_unit
from rdkit import Chem

from openff.interchange import Interchange
from openff.interchange.components.smirnoff import library_charge_from_molecule
from openff.interchange.drivers.openmm import _get_openmm_energies, get_openmm_energies
from openff.interchange.drivers.report import EnergyError
from openff.interchange.tests import (
    _compare_nonbonded_parameters,
    _compare_nonbonded_settings,
    _compare_torsion_forces,
    _get_force,
    get_test_file_path,
)

kj_mol = unit.kilojoule / unit.mol
_parsley = ForceField("openff-1.0.0.offxml")
_box_vectors = unit.Quantity(
    4 * np.eye(3),
    unit.nanometer,
)


def compare_single_mol_systems(mol, force_field):

    top = mol.to_topology()
    top.box_vectors = _box_vectors

    try:
        toolkit_sys = force_field.create_openmm_system(
            top,
            charge_from_molecules=[mol],
        )
    except (
        UnassignedBondParameterException,
        UnassignedAngleParameterException,
        UnassignedProperTorsionParameterException,
    ):
        pytest.xfail(f"Molecule failed! (missing valence parameters)\t{mol.to_inchi()}")

    toolkit_energy = _get_openmm_energies(
        toolkit_sys, box_vectors=_box_vectors, positions=mol.conformers[0]
    )

    openff_sys = Interchange.from_smirnoff(force_field=force_field, topology=top)
    openff_sys.positions = mol.conformers[0]
    system_energy = get_openmm_energies(openff_sys, combine_nonbonded_forces=True)

    toolkit_energy.compare(
        system_energy,
        custom_tolerances={
            "Bond": 1e-6 * kj_mol,
            "Angle": 1e-6 * kj_mol,
            "Torsion": 4e-5 * kj_mol,
            "Nonbonded": 1e-5 * kj_mol,
        },
    )


def compare_condensed_systems(mol, force_field):
    from openff.evaluator import unit as evaluator_unit  # type: ignore[import]
    from openff.evaluator.utils.packmol import pack_box  # type: ignore[import]

    mass_density = 500 * evaluator_unit.kilogram / evaluator_unit.meter ** 3

    trj, assigned_residue_names = pack_box(
        molecules=[mol], number_of_copies=[100], mass_density=mass_density
    )

    try:
        openff_top = Topology.from_openmm(trj.top.to_openmm(), unique_molecules=[mol])
    except ValueError:
        print(f"Molecule failed! (conversion from OpenMM)\t{mol.to_inchi()}")
        return

    box_vectors = trj.unitcell_vectors[0] * openmm_unit.nanometer
    openff_top.box_vectors = box_vectors

    try:
        toolkit_sys = force_field.create_openmm_system(
            openff_top,
            charge_from_molecules=[mol],
        )

    except (
        UnassignedBondParameterException,
        UnassignedAngleParameterException,
        UnassignedProperTorsionParameterException,
    ):
        print(f"Molecule failed! (missing valence parameters)\t{mol.to_inchi()}")
        return

    positions = trj.xyz[0] * openmm_unit.nanometer
    toolkit_energy = _get_openmm_energies(
        toolkit_sys,
        box_vectors=box_vectors,
        positions=positions,
    )

    openff_sys = Interchange.from_smirnoff(force_field=force_field, topology=openff_top)
    openff_sys.box = box_vectors
    openff_sys.positions = trj.xyz[0] * unit.nanometer

    new_sys = openff_sys.to_openmm(combine_nonbonded_forces=True)

    system_energy = _get_openmm_energies(
        new_sys,
        box_vectors=box_vectors,
        positions=positions,
    )

    # Where energies to not precisely match, inspect all parameters in each force
    try:
        toolkit_energy.compare(
            system_energy,
            custom_tolerances={
                "Bond": 1e-6 * kj_mol,
                "Angle": 1e-6 * kj_mol,
                "Torsion": 4e-5 * kj_mol,
                "Nonbonded": 1e-5 * kj_mol,
            },
        )
    except EnergyError as e:
        if "Torsion" in str(e):
            _compare_torsion_forces(
                _get_force(toolkit_sys, openmm.PeriodicTorsionForce),
                _get_force(new_sys, openmm.PeriodicTorsionForce),
            )
        if "Nonbonded" in str(e):
            _compare_nonbonded_settings(
                _get_force(toolkit_sys, openmm.NonbondedForce),
                _get_force(new_sys, openmm.NonbondedForce),
            )
            _compare_nonbonded_parameters(
                _get_force(toolkit_sys, openmm.NonbondedForce),
                _get_force(new_sys, openmm.NonbondedForce),
            )
        if "Bond" in str(e):
            raise e
        if "Angle" in str(e):
            raise e


@pytest.mark.skip(
    "Needs an OpenFF Evaluator release and/or refactoring out packing code to here"
)
@requires_openeye
@skip_if_missing("openff.evaluator")
@pytest.mark.timeout(120)
@pytest.mark.slow()
@pytest.mark.parametrize(
    "rdmol",
    Chem.SDMolSupplier(get_test_file_path("MiniDrugBankTrimmed.sdf"), sanitize=False),
)
def test_energy_vs_toolkit(rdmol):

    Chem.SanitizeMol(rdmol)
    mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    if mol.name in ["DrugBank_6182"]:
        pytest.xfail("OpenEye stereochemistry assumptions fail")
    elif mol.name in [
        "DrugBank_2148",
        "DrugBank_1971",
        "DrugBank_2563",
        "DrugBank_2585",
    ]:
        pytest.xfail(
            "These molecules results in small non-bonded energy differences on CI "
            "runners, although some behavior is observed on different hardware."
        )

    assert mol.n_conformers > 0

    max_n_heavy_atoms = 12
    if len([a for a in mol.atoms if a.atomic_number > 1]) > max_n_heavy_atoms:
        pytest.skip(f"Skipping > {max_n_heavy_atoms} heavy atoms for now")

    # Skip molecules that cause hangs due to the toolkit taking an excessive time
    # to match the library charge SMIRKS.
    if mol.to_inchikey(fixed_hydrogens=True) in [
        "MUUSFMCMCWXMEM-UHFFFAOYNA-N",  # CHEMBL1362008
    ]:

        pytest.skip(
            "toolkit taking an excessive time to match the library charge SMIRKS"
        )

    # Faster to load once and deepcopy N times than load N times
    parsley = deepcopy(_parsley)

    if mol.partial_charges is None:
        pytest.skip("missing partial charges")
    # mol.assign_partial_charges(partial_charge_method="am1bcc")

    # Avoid AM1BCC calulations by using the partial charges in the SDF file
    library_charge_handler = LibraryChargeHandler(version=0.3)
    library_charges = library_charge_from_molecule(mol)
    library_charge_handler.add_parameter(parameter=library_charges)
    parsley.register_parameter_handler(library_charge_handler)

    compare_condensed_systems(mol, parsley)
    parsley.deregister_parameter_handler(parsley["Constraints"])

    compare_single_mol_systems(mol, parsley)
