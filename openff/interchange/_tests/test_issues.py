"""Tests reproducing specific issues that are otherwise uncategorized."""

import parmed
from openff.toolkit import ForceField, Molecule, Topology
from openff.utilities import get_data_file_path

from openff.interchange._tests import MoleculeWithConformer


def test_issue_723():
    force_field = ForceField("openff-2.1.0.offxml")

    molecule = Molecule.from_smiles("C#N")
    molecule.generate_conformers(n_conformers=1)

    force_field.create_interchange(molecule.to_topology()).to_top("_x.top")

    parmed.load_file("_x.top")


def test_issue_1022():

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

    interchange = ForceField(
        "openff-2.0.0.offxml",
        get_data_file_path(
            "example-sigma-hole-bromine.offxml",
            "openff.interchange._tests.data",
        ),
    ).create_interchange(topology)

    interchange.to_top("tmp")
