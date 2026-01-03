"""Tests reproducing specific issues that are otherwise uncategorized."""

import random

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.utilities import get_data_file_path, has_executable, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_gmx, shuffle_topology
from openff.interchange.components._packmol import UNIT_CUBE, pack_box, solvate_topology
from openff.interchange.drivers import get_amber_energies, get_openmm_energies
from openff.interchange.exceptions import NonperiodicNoCutoffNotSupportedError


def test_issue_723():
    parmed = pytest.importorskip("parmed")
    force_field = ForceField("openff-2.1.0.offxml")

    molecule = Molecule.from_smiles("C#N")
    molecule.generate_conformers(n_conformers=1)

    force_field.create_interchange(molecule.to_topology()).to_top("_x.top")

    parmed.load_file("_x.top")


@pytest.mark.skipif(not has_executable("packmol"), reason="Packmol is not installed")
@pytest.mark.parametrize("pack", [True, False])
def test_issue_1022(pack):
    topology = Topology.from_molecules(
        [
            MoleculeWithConformer.from_smiles(smi)
            for smi in [
                "CBr",
                "O",
                "O",
                "O",
                "[Na+]",
                "[Cl-]",
            ]
        ],
    )

    topology.box_vectors = Quantity(numpy.eye(3) * 10.0, "nanometer")

    if pack:
        topology = pack_box(
            molecules=[*topology.unique_molecules],
            number_of_copies=[1, 3, 1, 1],
            box_vectors=topology.box_vectors,
        )

    force_field = ForceField(
        "openff-2.0.0.offxml",
        get_data_file_path(
            "example-sigma-hole-bromine.offxml",
            "openff.interchange._tests.data",
        ),
    )

    interchange = force_field.create_interchange(topology)

    interchange.to_top("tmp")

    if pack:
        for seed in random.sample(range(0, 10**10), 5):
            # TODO: Compare GROMACS energies here as well
            get_openmm_energies(interchange).compare(
                get_openmm_energies(
                    shuffle_topology(
                        interchange,
                        force_field,
                        seed=seed,
                    ),
                ),
                tolerances={"Nonbonded": Quantity("1e-3 kilojoule_per_mole")},
            )


@skip_if_missing("openmm")
def test_issue_1031():
    import openmm.app

    # just grab some small PDB file from the toolkit, doesn't need to be huge, just
    # needs to include some relevant atom names
    openmm_topology = openmm.app.PDBFile(
        get_data_file_path(
            "proteins/MainChain_HIE.pdb",
            "openff.toolkit",
        ),
    ).topology

    openmm_atom_names = {atom.name for atom in openmm_topology.atoms()}

    interchange = Interchange.from_openmm(
        system=openmm.app.ForceField(
            "amber99sb.xml",
            "tip3p.xml",
        ).createSystem(
            openmm_topology,
            nonbondedMethod=openmm.app.PME,
        ),
        topology=openmm_topology,
    )

    openff_atom_names = {atom.name for atom in interchange.topology.atoms}

    assert sorted(openmm_atom_names) == sorted(openff_atom_names)

    # check a few atom names to ensure these didn't end up being empty sets
    for atom_name in ("NE2", "H3", "HA", "CH3", "CA", "CB", "CE1"):
        assert atom_name in openff_atom_names


def test_issue_1049():
    pytest.importorskip("openmm")

    topology = Topology.from_molecules(
        [
            Molecule.from_smiles("C"),
            Molecule.from_smiles("O"),
            Molecule.from_smiles("O"),
        ],
    )

    interchange = ForceField("openff-2.2.0.offxml", "opc.offxml").create_interchange(topology)

    openmm_topology = interchange.to_openmm_topology()
    openmm_system = interchange.to_openmm_system()

    # the same index in system should also be a virtual site in the topology
    for particle_index, particle in enumerate(openmm_topology.atoms()):
        assert openmm_system.isVirtualSite(particle_index) == (particle.element is None), (
            f"particle index {particle_index} is a virtual site in the system OR topology but not both"
        )


def test_issue_1052(sage, ethanol):
    """Test that _SMIRNOFFElectrostaticsCollection.charges is populated."""
    out = sage.create_interchange(ethanol.to_topology())

    assert len(out["Electrostatics"].charges) > 0


@skip_if_missing("openmm")
def test_issue_1209(sage, ethanol):
    from openff.interchange.drivers.openmm import get_openmm_energies

    ethanol.assign_partial_charges("gasteiger")
    ethanol.generate_conformers(n_conformers=1)

    out = sage.create_interchange(ethanol.to_topology(), charge_from_molecules=[ethanol])

    get_openmm_energies(out, combine_nonbonded_forces=False).compare(
        get_openmm_energies(
            Interchange.model_validate_json(out.model_dump_json()),
            combine_nonbonded_forces=False,
        ),
    )


def test_issue_1337(water):
    """Reproduce Issue #1337."""
    ff14sb_tip3p = ForceField("ff14sb_off_impropers_0.0.3.offxml", "tip3p.offxml")

    # just make sure this doesn't crash (original issue)
    # more extensive tests in openff/interchange/_tests/unit_tests/interop/openmm/test_constraints.py
    ff14sb_tip3p.create_interchange(water.to_topology())


@needs_gmx
def test_issue_1361_gromacs(caffeine, sage, tmp_path):
    """Test that the 'how to opt in to pseudo-vacuum' message is communicated."""
    interchange = sage.create_interchange(caffeine.to_topology())

    with pytest.raises(
        NonperiodicNoCutoffNotSupportedError,
        match=r"GROMACS versions 2020 and newer do not support systems without periodicity",
    ):
        interchange.to_gromacs(prefix="foo")


@pytest.mark.skipif(not has_executable("sander"), reason="sander not installed")
def test_issue_1361_amber(caffeine, sage, tmp_path):
    """Test that the 'how to opt in to pseudo-vacuum' message is communicated."""
    interchange = sage.create_interchange(caffeine.to_topology())

    with pytest.raises(
        NonperiodicNoCutoffNotSupportedError,
        match=r"vdW method no-cutoff not supported",
    ):
        interchange.to_amber(prefix="bar")


@pytest.mark.skipif(not has_executable("sander"), reason="sander not installed")
def test_issue_1395_amber(caffeine, sage, water, tmp_path):
    """Test that water charges are correctly ordered in ligand + water systems to Amber."""

    # turn off switching to make energy comparisons easier
    sage["vdW"].switch_width = Quantity(0.0, "nanometer")

    parmed = pytest.importorskip("parmed")

    topology = solvate_topology(
        topology=caffeine.to_topology(),
        nacl_conc=Quantity("0.0 mole/liter"),
        padding=Quantity("1.0 nanometer"),
        box_shape=UNIT_CUBE,
        target_density=Quantity("0.7 gram / centimeter**3"),
    )

    interchange = sage.create_interchange(topology)

    interchange.to_amber("solvated_caffeine")

    structure = parmed.load_file("solvated_caffeine.prmtop")

    # these should all be oxygens
    assert set([a.charge for a in structure.atoms][caffeine.n_atoms :: 3]) == {-0.834}

    # these should all be hydrogens
    assert set([a.charge for a in structure.atoms][caffeine.n_atoms + 1 :: 3]) == {0.417}
    assert set([a.charge for a in structure.atoms][caffeine.n_atoms + 2 :: 3]) == {0.417}

    get_openmm_energies(interchange, combine_nonbonded_forces=False).compare(
        get_amber_energies(interchange),
        tolerances={
            "Angle": Quantity("1e10 kilojoule_per_mole"),  # constraints weirdness
            "vdW": Quantity("0.1 kilojoule_per_mole"),
            "Electrostatics": Quantity("0.5 kilojoule_per_mole"),
        },
    )
