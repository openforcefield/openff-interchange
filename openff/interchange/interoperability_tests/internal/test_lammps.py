import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import _OFFBioTop
from openff.interchange.drivers import get_lammps_energies, get_openmm_energies
from openff.interchange.drivers.lammps import _write_lammps_input
from openff.interchange.testing.utils import needs_lmp


@needs_lmp
@pytest.mark.slow()
@pytest.mark.parametrize("n_mols", [1, 2])
@pytest.mark.parametrize(
    "mol",
    [
        "C",
        "CC",  # Adds a proper torsion term(s)
        "C=O",  # Simplest molecule with any improper torsion
        "OC=O",  # Simplest molecule with a multi-term torsion
        "CCOC",  # This hits t86, which has a non-1.0 idivf
        "C1COC(=O)O1",  # This adds an improper, i2
    ],
)
def test_to_lammps_single_mols(mol, n_mols):
    """
    Test that Interchange.to_openmm Interchange.to_lammps report sufficiently similar energies.

    TODO: Tighten tolerances
    TODO: Test periodic and non-periodic
    """

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    mol = Molecule.from_smiles(mol)
    mol.generate_conformers(n_conformers=1)
    tmp = Topology.from_molecules(n_mols * [mol])
    top = _OFFBioTop.from_molecules(
        mdtop=md.Topology.from_openmm(tmp.to_openmm()), molecules=n_mols * [mol]
    )
    mol.conformers[0] -= np.min(mol.conformers) * omm_unit.angstrom

    top.box_vectors = np.eye(3) * np.asarray([10, 10, 10]) * omm_unit.nanometer

    if n_mols == 1:
        positions = mol.conformers[0]
    elif n_mols == 2:
        positions = np.vstack(
            [mol.conformers[0], mol.conformers[0] + 3 * omm_unit.nanometer]
        )
        positions = positions * omm_unit.angstrom

    openff_sys = Interchange.from_smirnoff(parsley, top)
    openff_sys.top.mdtop = md.Topology.from_openmm(top.to_openmm())
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
            "Torsion": 3e-5 * omm_unit.kilojoule_per_mole,
        },
    )
