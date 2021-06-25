import tempfile

import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.utilities import temporary_cd
from simtk import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.gromacs import _get_mdp_file, _run_gmx_energy
from openff.interchange.tests.energy_tests.test_energies import needs_gmx
from openff.interchange.types import ArrayQuantity


def openff_openmm_pmd_gmx(
    topology: Topology,
    forcefield: ForceField,
    box: ArrayQuantity,
    prefix: str,
) -> None:
    """Pipeline to write GROMACS files from and OpenMM interchange through ParmEd"""
    topology.box_vectors = box.to(unit.nanometer).magnitude * omm_unit.nanometer  # type: ignore
    omm_sys = forcefield.create_openmm_system(topology)

    struct = pmd.openmm.load_topology(
        system=omm_sys,
        topology=topology.to_openmm(),
        xyz=topology.topology_molecules[0].reference_molecule.conformers[0],
    )

    # Assign dummy residue names, GROMACS will not accept empty strings
    # TODO: Patch upstream?
    for res in struct.residues:
        res.name = "FOO"

    struct.save(prefix + ".gro")
    struct.save(prefix + ".top")


def openff_pmd_gmx_indirect(
    topology: Topology,
    forcefield: ForceField,
    box: ArrayQuantity,
    prefix: str,
) -> None:
    off_sys = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
    off_sys.box = box

    ref_mol = topology.topology_molecules[0].reference_molecule
    off_top_positions = ref_mol.conformers[0]
    # TODO: Update this when better processing of OFFTop positions is supported
    off_sys.positions = off_top_positions

    struct = off_sys._to_parmed()

    struct.save(prefix + ".gro")
    struct.save(prefix + ".top")


def openff_pmd_gmx_direct(
    topology: Topology,
    forcefield: ForceField,
    box: ArrayQuantity,
    prefix: str,
) -> None:
    off_sys = Interchange.from_smirnoff(forcefield, topology=topology)
    off_sys.box = box

    ref_mol = topology.topology_molecules[0].reference_molecule
    off_top_positions = ref_mol.conformers[0]
    # TODO: Update this when better processing of OFFTop positions is supported
    off_sys.positions = off_top_positions
    off_sys.positions = np.round(off_sys.positions, 3)

    off_sys.to_gro(prefix + ".gro")
    off_sys.to_top(prefix + ".top")


@needs_gmx
@pytest.mark.slow
@pytest.mark.parametrize("smiles", ["C"])
def test_parmed_openmm(tmpdir, smiles):
    tmpdir.chdir()

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(mol)
    box = 4 * np.eye(3) * unit.nanometer

    with tempfile.TemporaryDirectory() as omm_tempdir:
        with temporary_cd(omm_tempdir):
            openff_openmm_pmd_gmx(
                topology=top,
                forcefield=parsley,
                box=box,
                prefix="via_openmm",
            )

            ener1 = _run_gmx_energy(
                top_file="via_openmm.top",
                gro_file="via_openmm.gro",
                mdp_file=_get_mdp_file("cutoff"),
            )

    with tempfile.TemporaryDirectory() as off_tempdir:
        with temporary_cd(off_tempdir):
            openff_pmd_gmx_indirect(
                topology=top,
                forcefield=parsley,
                box=box,
                prefix="via_conversion",
            )

            ener2 = _run_gmx_energy(
                top_file="via_conversion.top",
                gro_file="via_conversion.gro",
                mdp_file=_get_mdp_file("cutoff"),
            )

    ener1.compare(ener2)

    with tempfile.TemporaryDirectory() as off_tempdir:
        with temporary_cd(off_tempdir):
            openff_pmd_gmx_direct(
                topology=top,
                forcefield=parsley,
                box=box,
                prefix="via_call",
            )

            ener3 = _run_gmx_energy(
                top_file="via_call.top",
                gro_file="via_call.gro",
                mdp_file=_get_mdp_file("cutoff"),
            )

    ener2.compare(
        ener3,
        custom_tolerances={
            "Bond": 1.0 * omm_unit.kilojoule_per_mole,
            "Angle": 0.22 * omm_unit.kilojoule_per_mole,
        },
    )
