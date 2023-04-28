"""
Units tests for openff.interchange.components._packmol
"""
import numpy as np
import pytest
from openff.units import unit

from openff.interchange.components._packmol import pack_box
from openff.interchange.exceptions import PACKMOLRuntimeError, PACKMOLValueError


def test_packmol_box_size():
    from openff.toolkit.topology import Molecule

    molecules = [Molecule.from_smiles("O")]

    trajectory, _ = pack_box(
        molecules,
        [10],
        box_size=([20] * 3) * unit.angstrom,
    )

    assert trajectory is not None

    assert trajectory.n_chains == 1
    assert trajectory.n_residues == 10
    assert trajectory.n_atoms == 30
    assert trajectory.topology.n_bonds == 20

    assert all(x.name == "HOH" for x in trajectory.top.residues)

    assert np.allclose(trajectory.unitcell_lengths, 2.2)


def test_packmol_bad_input():
    from openff.toolkit.topology import Molecule

    molecules = [Molecule.from_smiles("O")]

    with pytest.raises(PACKMOLValueError):
        pack_box(molecules, [10, 20], box_size=([20] * 3) * unit.angstrom)


def test_packmol_failed():
    from openff.toolkit.topology import Molecule

    molecules = [Molecule.from_smiles("O")]

    with pytest.raises(PACKMOLRuntimeError):
        pack_box(molecules, [10], box_size=([0.1] * 3) * unit.angstrom)


def test_packmol_water():
    from openff.toolkit.topology import Molecule

    molecules = [Molecule.from_smiles("O")]

    trajectory, _ = pack_box(
        molecules,
        [10],
        mass_density=1.0 * unit.grams / unit.milliliters,
    )

    assert trajectory is not None

    assert trajectory.n_chains == 1
    assert trajectory.n_residues == 10
    assert trajectory.n_atoms == 30
    assert trajectory.topology.n_bonds == 20

    assert all(x.name == "HOH" for x in trajectory.top.residues)


def test_packmol_ions():
    from openff.toolkit.topology import Molecule

    molecules = [
        Molecule.from_smiles("[Na+]"),
        Molecule.from_smiles("[Cl-]"),
        Molecule.from_smiles("[K+]"),
    ]

    trajectory, _ = pack_box(
        molecules,
        [1, 1, 1],
        box_size=([20] * 3) * unit.angstrom,
    )

    assert trajectory is not None

    assert trajectory.n_chains == 3
    assert trajectory.n_residues == 3
    assert trajectory.n_atoms == 3
    assert trajectory.topology.n_bonds == 0

    assert trajectory.top.residue(0).name == "Na+"
    assert trajectory.top.residue(1).name == "Cl-"
    assert trajectory.top.residue(2).name == "K+"

    assert trajectory.top.atom(0).name == "Na+"
    assert trajectory.top.atom(1).name == "Cl-"
    assert trajectory.top.atom(2).name == "K+"


def test_packmol_paracetamol():
    from openff.toolkit.topology import Molecule

    # Test something a bit more tricky than water
    molecules = [Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")]

    trajectory, _ = pack_box(
        molecules,
        [1],
        box_size=([20] * 3) * unit.angstrom,
    )

    assert trajectory is not None

    assert trajectory.n_chains == 1
    assert trajectory.n_residues == 1
    assert trajectory.n_atoms == 20
    assert trajectory.topology.n_bonds == 20


def test_amino_acids():
    amino_residues = {
        "C[C@H](N)C(=O)O": "ALA",
        # Undefined stereochemistry error.
        # "N=C(N)NCCC[C@H](N)C(=O)O": "ARG",
        "NC(=O)C[C@H](N)C(=O)O": "ASN",
        "N[C@@H](CC(=O)O)C(=O)O": "ASP",
        "N[C@@H](CS)C(=O)O": "CYS",
        "N[C@@H](CCC(=O)O)C(=O)O": "GLU",
        "NC(=O)CC[C@H](N)C(=O)O": "GLN",
        "NCC(=O)O": "GLY",
        "N[C@@H](Cc1c[nH]cn1)C(=O)O": "HIS",
        "CC[C@H](C)[C@H](N)C(=O)O": "ILE",
        "CC(C)C[C@H](N)C(=O)O": "LEU",
        "NCCCC[C@H](N)C(=O)O": "LYS",
        "CSCC[C@H](N)C(=O)O": "MET",
        "N[C@@H](Cc1ccccc1)C(=O)O": "PHE",
        "O=C(O)[C@@H]1CCCN1": "PRO",
        "N[C@@H](CO)C(=O)O": "SER",
        "C[C@@H](O)[C@H](N)C(=O)O": "THR",
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": "TRP",
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O": "TYR",
        "CC(C)[C@H](N)C(=O)O": "VAL",
    }

    smiles = [*amino_residues]

    from openff.toolkit.topology import Molecule

    molecules = [Molecule.from_smiles(x) for x in smiles]
    counts = [1] * len(smiles)

    trajectory, _ = pack_box(
        molecules,
        counts,
        box_size=([1000] * 3) * unit.angstrom,
    )

    assert trajectory is not None

    assert trajectory.n_chains == len(smiles)
    assert trajectory.n_residues == len(smiles)

    for index, _smiles in enumerate(smiles):
        assert trajectory.top.residue(index).name == amino_residues[_smiles]
