"""Tests reproducing specific issues that are otherwise uncategorized."""

import random

import numpy
import parmed
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.utilities import get_data_file_path

from openff.interchange._tests import MoleculeWithConformer, shuffle_topology
from openff.interchange.components._packmol import pack_box
from openff.interchange.drivers import get_openmm_energies


def test_issue_723():
    force_field = ForceField("openff-2.1.0.offxml")

    molecule = Molecule.from_smiles("C#N")
    molecule.generate_conformers(n_conformers=1)

    force_field.create_interchange(molecule.to_topology()).to_top("_x.top")

    parmed.load_file("_x.top")


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
