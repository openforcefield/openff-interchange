"""
Units tests for openff.interchange.components._packmol
"""
import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.interchange.components._packmol import pack_box
from openff.interchange.exceptions import PACKMOLRuntimeError, PACKMOLValueError


@pytest.fixture(scope="module")
def molecules() -> list[Molecule]:
    return [Molecule.from_smiles("O")]


def test_packmol_box_vectors(molecules):
    topology = pack_box(
        molecules,
        [10],
        box_vectors=20 * numpy.identity(3) * unit.angstrom,
    )

    assert topology is not None

    assert topology.n_atoms == 30
    assert topology.n_bonds == 20

    numpy.testing.assert_allclose(
        topology.box_vectors.m_as(unit.nanometer).diagonal(),
        [2.0, 2.0, 2.0],
    )


def test_packmol_bad_copies(molecules):
    with pytest.raises(PACKMOLValueError):
        pack_box(
            molecules,
            [10, 20],
            box_vectors=20 * numpy.identity(3) * unit.angstrom,
        )


def test_packmol_bad_box_vectors(molecules):
    with pytest.raises(PACKMOLValueError, match=r"with shape \(3, 3\)"):
        pack_box(
            molecules,
            [2],
            box_vectors=20 * numpy.identity(4) * unit.angstrom,
        )


def test_packmol_underspecified(molecules):
    """Too few arguments are provided."""

    with pytest.raises(PACKMOLValueError, match="One of.*must be"):
        pack_box(
            molecules,
            number_of_copies=[1],
        )


def test_packmol_overspecified(molecules):
    """Too many arguments are provided."""

    with pytest.raises(PACKMOLValueError, match="cannot be specified together"):
        pack_box(
            molecules,
            number_of_copies=[1],
            mass_density=1.0 * unit.grams / unit.milliliter,
            box_vectors=20 * numpy.identity(3) * unit.angstrom,
        )


def test_packmol_bad_solute(molecules):
    with pytest.raises(PACKMOLValueError):
        pack_box(
            molecules,
            [2],
            box_vectors=20 * numpy.identity(3) * unit.angstrom,
            solute="my_solute.pdb",
        )


def test_packmol_failed(molecules):
    with pytest.raises(PACKMOLRuntimeError):
        pack_box(
            molecules,
            [10],
            box_vectors=0.1 * numpy.identity(3) * unit.angstrom,
        )


def test_packmol_water(molecules):
    topology = pack_box(
        molecules,
        [10],
        mass_density=1.0 * unit.grams / unit.milliliter,
    )

    assert topology is not None

    assert topology.n_atoms == 30
    assert topology.n_bonds == 20
    assert topology.n_molecules == 10


def test_packmol_ions():
    molecules = [
        Molecule.from_smiles("[Na+]"),
        Molecule.from_smiles("[Cl-]"),
        Molecule.from_smiles("[K+]"),
    ]

    topology = pack_box(
        molecules,
        [1, 1, 1],
        box_vectors=20 * numpy.identity(3) * unit.angstrom,
    )

    assert topology is not None

    assert topology.n_atoms == 3
    assert topology.n_bonds == 0

    # Na+
    assert topology.atom(0).formal_charge == +1 * unit.elementary_charge
    assert topology.atom(0).atomic_number == 11

    # Cl-
    assert topology.atom(1).formal_charge == -1 * unit.elementary_charge
    assert topology.atom(1).atomic_number == 17
    # K+
    assert topology.atom(2).formal_charge == +1 * unit.elementary_charge
    assert topology.atom(2).atomic_number == 19


def test_packmol_paracetamol():
    # Test something a bit more tricky than water
    molecules = [Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")]

    topology = pack_box(
        molecules,
        [1],
        box_vectors=20 * numpy.identity(3) * unit.angstrom,
    )

    assert topology is not None

    assert topology.n_atoms == 20
    assert topology.n_bonds == 20


@pytest.mark.slow()
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

    molecules = [Molecule.from_smiles(x) for x in smiles]
    counts = [1] * len(smiles)

    topology = pack_box(
        molecules,
        counts,
        box_vectors=1000 * numpy.identity(3) * unit.angstrom,
    )

    assert topology is not None


def test_pack_diatomic_ion():
    molecules = [Molecule.from_smiles("[Mg+2]"), Molecule.from_smiles("[Cl-]")]

    topology = pack_box(
        molecules,
        [1, 2],
        box_vectors=20 * numpy.identity(3) * unit.angstrom,
    )

    assert topology is not None

    assert topology.atom(0).atomic_number == 12
    assert topology.n_atoms == 3

    numpy.testing.assert_allclose(
        topology.box_vectors.m_as(unit.nanometer).diagonal(),
        [2.0, 2.0, 2.0],
    )


def test_solvate_structure():
    benzene = Molecule.from_smiles("c1ccccc1")

    with pytest.raises(
        PACKMOLValueError,
        match="missing some atomic positions",
    ):
        pack_box(
            [Molecule.from_smiles("O")],
            [10],
            box_vectors=50 * numpy.identity(3) * unit.angstrom,
            solute=benzene.to_topology(),
        )

    benzene.generate_conformers(n_conformers=1)

    topology = pack_box(
        [Molecule.from_smiles("O")],
        [10],
        box_vectors=50 * numpy.identity(3) * unit.angstrom,
        solute=benzene.to_topology(),
    )

    assert topology.n_molecules == 11
    assert len([*topology.unique_molecules]) == 2


def solvate_topology():
    ligand = Molecule.from_smiles("C1CN2C(=N1)SSC2=S")
    ligand.generate_conformers(n_conformers=1)

    solvated_topology = solvate_topology(
        ligand.to_topology(),
    )

    assert solvated_topology.molecule(0).to_smiles() == ligand.to_smiles()

    assert solvated_topology.molecule(0).to_smiles(explicit_hydrogens=False) == "O"

    assert solvated_topology.molecule(
        solvated_topology.n_molecules - 1,
    ).to_smiles() in ["[Cl-]", "[Na+]"]

    # Defaults should produce something like 595, but just sanity check here
    assert solvated_topology.n_molecules > 500

    for position in [
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 2],
    ]:
        assert solvated_topology.box_vectors[position].m == 0.0

    assert solvated_topology.box_vectors[0, 0] == solvated_topology.box_vectors[1, 1]
    assert solvated_topology.box_vectors[2, 0] == solvated_topology.box_vectors[2, 1]
