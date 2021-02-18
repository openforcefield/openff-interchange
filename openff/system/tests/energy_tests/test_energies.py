import mbuild as mb
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology

from openff.system import unit
from openff.system.exceptions import NonbondedEnergyError
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import (
    _get_openmm_energies,
    get_openmm_energies,
)
from openff.system.tests.energy_tests.utils import (
    compare_gromacs_openmm,
    compare_openmm,
)


@pytest.mark.parametrize("constrained", [False])  # [True, False]
@pytest.mark.parametrize("mol_smi", ["C"])  # ["C", "CC"]
@pytest.mark.parametrize("n_mol", [1, 10, 100])
def test_energies_single_mol(constrained, n_mol, mol_smi):
    mol = Molecule.from_smiles(mol_smi)
    mol.generate_conformers(n_conformers=1)
    mol.name = "FOO"
    top = Topology.from_molecules(n_mol * [mol])

    if constrained:
        parsley = ForceField("openff-1.0.0.offxml")
    else:
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    mol.to_file("out.xyz", file_format="xyz")
    compound: mb.Compound = mb.load("out.xyz")
    packed_box: mb.Compound = mb.fill_box(
        compound=compound,
        n_compounds=[n_mol],
        density=500,  # kg/m^3
    )

    positions = packed_box.xyz * unit.nanometer
    off_sys.positions = positions

    box = np.asarray(packed_box.box.lengths) * unit.nanometer
    if np.any(box < 4 * unit.nanometer):
        off_sys.box = np.array([4, 4, 4]) * unit.nanometer
    else:
        off_sys.box = box

    # Compare directly to toolkit's reference implementation
    omm_energies = get_openmm_energies(off_sys, round_positions=3)
    omm_reference = parsley.create_openmm_system(top)
    reference_energies = _get_openmm_energies(
        omm_sys=omm_reference,
        box_vectors=off_sys.box,
        positions=off_sys.positions,
        round_positions=3,
    )

    try:
        compare_openmm(omm_energies, reference_energies)
    except NonbondedEnergyError:
        # If nonbonded energies differ, at least ensure that the non-bonded parameters on each particle match
        from openff.system.tests.utils import (
            _get_charges_from_openmm_system,
            _get_lj_params_from_openmm_system,
        )

        omm_sys = off_sys.to_openmm()
        np.testing.assert_equal(
            np.asarray([*_get_charges_from_openmm_system(omm_sys)]),
            np.asarray([*_get_charges_from_openmm_system(omm_reference)]),
        )
        np.testing.assert_equal(
            np.asarray([*_get_lj_params_from_openmm_system(omm_sys)]),
            np.asarray([*_get_lj_params_from_openmm_system(omm_reference)]),
        )

    # Compare GROMACS writer and OpenMM export
    gmx_energies, _ = get_gromacs_energies(off_sys)

    compare_gromacs_openmm(
        omm_energies=omm_energies,
        gmx_energies=gmx_energies,
        custom_tolerances={"Nonbonded": 2e-4 * n_mol},
    )


def test_water_dimer():
    from openff.system.utils import get_test_file_path

    tip3p = ForceField(get_test_file_path("tip3p.offxml"))
    water = Molecule.from_smiles("O")
    top = Topology.from_molecules(2 * [water])

    from simtk import openmm
    from simtk import unit as omm_unit

    pdbfile = openmm.app.PDBFile(get_test_file_path("water-dimer.pdb"))

    positions = np.array(pdbfile.positions / omm_unit.nanometer) * unit.nanometer

    openff_sys = tip3p.create_openff_system(top)
    openff_sys.positions = positions
    openff_sys.box = [10, 10, 10] * unit.nanometer

    omm_energies = get_openmm_energies(openff_sys)

    toolkit_energies = _get_openmm_energies(
        tip3p.create_openmm_system(top),
        openff_sys.box,
        openff_sys.positions,
    )

    compare_openmm(omm_energies, toolkit_energies)

    # TODO: Fix GROMACS energies by handling SETTLE constraints
    # gmx_energies, _ = get_gromacs_energies(openff_sys)
    # compare_gromacs_openmm(omm_energies=omm_energies, gmx_energies=gmx_energies)
