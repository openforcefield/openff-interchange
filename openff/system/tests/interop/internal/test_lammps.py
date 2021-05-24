import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.components.mdtraj import OFFBioTop
from openff.system.drivers import get_lammps_energies, get_openmm_energies
from openff.system.drivers.lammps import _write_lammps_input
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.test_energies import needs_lmp


@needs_lmp
@pytest.mark.slow
@pytest.mark.parametrize("n_mols", [1, 2])
@pytest.mark.parametrize(
    "mol",
    [
        "C",
        "CC",  # Adds a proper torsion term(s)
        "C=O",  # Simplest molecule with any improper torsion
        pytest.param(
            "OC=O",
            marks=pytest.mark.xfail(reason="degenerate impropers"),
        ),  # Simplest molecule with a multi-term torsion
        "CCOC",  # This hits t86, which has a non-1.0 idivf
        pytest.param(
            "C1COC(=O)O1",
            marks=pytest.mark.xfail(reason="degenerate impropers"),
        ),  # This adds an improper, i2
    ],
)
def test_to_lammps_single_mols(mol, n_mols):
    """
    Test that ForceField.create_openmm_system and System.to_openmm produce
    objects with similar energies

    TODO: Tighten tolerances
    TODO: Test periodic and non-periodic
    """

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    top = OFFBioTop.from_molecules(n_mols * [mol])
    top.mdtop = md.Topology.from_openmm(top.to_openmm())
    mol.conformers[0] -= np.min(mol.conformers) * omm_unit.angstrom

    top.box_vectors = np.eye(3) * np.asarray([10, 10, 10]) * omm_unit.nanometer

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 3 * omm_unit.nanometer]
        )
        positions = positions * omm_unit.angstrom

    openff_sys = parsley.create_openff_system(topology=top)
    openff_sys.positions = positions.value_in_unit(omm_unit.nanometer)
    openff_sys.box = top.box_vectors

    reference = get_openmm_energies(
        off_sys=openff_sys,
        round_positions=3,
    )

    lmp_energies = get_lammps_energies(
        off_sys=openff_sys,
        round_positions=3,
    )

    _write_lammps_input(
        off_sys=openff_sys,
        file_name="tmp.in",
    )

    lmp_energies.compare(
        reference,
        custom_tolerances={
            "Nonbonded": 100 * omm_unit.kilojoule_per_mole,
            "Electrostatics": 100 * omm_unit.kilojoule_per_mole,
            "vdW": 100 * omm_unit.kilojoule_per_mole,
            "Torsion": 0.005 * omm_unit.kilojoule_per_mole,
        },
    )
