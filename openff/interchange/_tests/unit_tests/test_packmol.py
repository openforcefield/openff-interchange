"""
Units tests for openff.interchange.packmol
"""

import pathlib

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.units import Quantity, unit
from openff.utilities import has_package, skip_if_missing

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.exceptions import PACKMOLRuntimeError, PACKMOLValueError
from openff.interchange.packmol import (
    RHOMBIC_DODECAHEDRON,
    RHOMBIC_DODECAHEDRON_XYHEX,
    UNIT_CUBE,
    _compute_brick_from_box_vectors,
    _find_packmol,
    _scale_box,
    pack_box,
    solvate_topology,
    solvate_topology_nonwater,
)


@pytest.fixture(scope="module")
def molecules(water) -> list[Molecule]:
    return [water]


@pytest.mark.skipif(_find_packmol() is None, reason="PACKMOL not found")
class TestPackmolWrapper:
    @pytest.mark.parametrize(
        "box",
        [
            UNIT_CUBE,
            RHOMBIC_DODECAHEDRON,
            RHOMBIC_DODECAHEDRON_XYHEX,
            50 * UNIT_CUBE,
            -UNIT_CUBE,
        ],
    )
    @pytest.mark.parametrize(
        "volume",
        [
            1.0 * unit.angstrom**3,
            1.0 * unit.nanometer**3,
            0.0 * unit.angstrom**3,
            234 * unit.angstrom**3,
        ],
    )
    def test_scale_box(self, box, volume):
        """Test that _scale_box() produces a box with the desired volume."""
        scaled_box = _scale_box(box, volume)
        a, b, c = scaled_box

        # | (a x b) . c | is the volume of the box
        # _scale_box uses numpy.linalg.det instead
        # linear dimensions are scaled by 1.1, so volumes are scaled by 1.1 ** 3
        assert numpy.isclose(numpy.abs(numpy.dot(numpy.cross(a, b), c)), volume * 1.1**3)

        assert scaled_box.u == unit.angstrom

    @pytest.mark.parametrize(
        "box",
        [
            UNIT_CUBE * unit.angstrom,
            RHOMBIC_DODECAHEDRON * unit.angstrom,
            RHOMBIC_DODECAHEDRON_XYHEX * unit.angstrom,
            UNIT_CUBE * unit.nanometer,
            50 * UNIT_CUBE * unit.angstrom,
            165.0 * RHOMBIC_DODECAHEDRON * unit.angstrom,
        ],
    )
    def test_compute_brick_from_box_vectors(self, box):
        """
        Test that _compute_brick() returns a rectangular box with the same volume
        and units.
        """
        brick = _compute_brick_from_box_vectors(box)
        # Same units
        assert brick.u == box.u
        # Same volume
        length, width, height = brick.m
        assert numpy.isclose(
            numpy.abs(numpy.linalg.det(box.m)),
            numpy.abs(length * width * height),
        )
        # Is rectangular
        assert brick.shape == (3,)

    def test_compute_brick_from_box_vectors_not_reduced(self):
        """
        Test that _compute_brick() raises an exception with an irreduced box.
        """
        # This is a rhombic dodecahedron with the first and last rows swapped
        box = Quantity(
            numpy.asarray(
                [
                    [0.5, 0.5, numpy.sqrt(2.0) / 2.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            ),
            unit.nanometer,
        )
        with pytest.raises(AssertionError):
            _compute_brick_from_box_vectors(box)

    def test_packmol_box_vectors(self, molecules):
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

    def test_packmol_bad_copies(self, molecules):
        with pytest.raises(PACKMOLValueError):
            pack_box(
                molecules,
                [10, 20],
                box_vectors=20 * numpy.identity(3) * unit.angstrom,
            )

    def test_packmol_bad_box_vectors(self, molecules):
        with pytest.raises(PACKMOLValueError, match=r"with shape \(3, 3\)"):
            pack_box(
                molecules,
                [2],
                box_vectors=20 * numpy.identity(4) * unit.angstrom,
            )

    def test_packmol_bad_box_shape(self, molecules):
        with pytest.raises(PACKMOLValueError, match=r"with shape \(3, 3\)"):
            solvate_topology(
                molecules[0].to_topology(),
                box_shape=20 * numpy.identity(4) * unit.angstrom,
            )

        with pytest.raises(PACKMOLValueError, match=r"with shape \(3, 3\)"):
            solvate_topology_nonwater(
                molecules[0].to_topology(),
                solvent=Molecule.from_smiles("CCCCCCO"),
                box_shape=20 * numpy.identity(4) * unit.angstrom,
                target_density=1.0 * unit.grams / unit.milliliter,
            )

    def test_packmol_underspecified(self, molecules):
        """Too few arguments are provided."""

        with pytest.raises(PACKMOLValueError, match="One of.*must be"):
            pack_box(
                molecules,
                number_of_copies=[1],
            )

    def test_packmol_overspecified(self, molecules):
        """Too many arguments are provided."""

        with pytest.raises(PACKMOLValueError, match="cannot be specified together"):
            pack_box(
                molecules,
                number_of_copies=[1],
                target_density=1.0 * unit.grams / unit.milliliter,
                box_vectors=20 * numpy.identity(3) * unit.angstrom,
            )

    def test_packmol_bad_solute(self, molecules):
        with pytest.raises(PACKMOLValueError):
            pack_box(
                molecules,
                [2],
                box_vectors=20 * numpy.identity(3) * unit.angstrom,
                solute="my_solute.pdb",
            )

    def test_packmol_failed(self, molecules):
        with pytest.raises(PACKMOLRuntimeError):
            pack_box(
                molecules,
                [10],
                box_vectors=0.1 * numpy.identity(3) * unit.angstrom,
            )

    def test_packmol_water(self, molecules):
        topology = pack_box(
            molecules,
            [10],
            target_density=1.0 * unit.grams / unit.milliliter,
        )

        assert topology is not None

        assert topology.n_atoms == 30
        assert topology.n_bonds == 20
        assert topology.n_molecules == 10

    def test_packmol_ions(self):
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

    def test_packmol_paracetamol(self):
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

    def test_amino_acids(self):
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

    def test_pack_diatomic_ion(self):
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

    @skip_if_missing("openmm")
    def test_solvate_structure(self, molecules):
        benzene = Molecule.from_smiles("c1ccccc1")

        with pytest.raises(
            PACKMOLValueError,
            match="missing some atomic positions",
        ):
            pack_box(
                molecules,
                [10],
                box_vectors=50 * numpy.identity(3) * unit.angstrom,
                solute=benzene.to_topology(),
            )

        benzene.generate_conformers(n_conformers=1)

        topology = pack_box(
            molecules,
            [10],
            box_vectors=50 * numpy.identity(3) * unit.angstrom,
            solute=benzene.to_topology(),
        )

        assert topology.n_molecules == 11
        assert len([*topology.unique_molecules]) == 2

    @skip_if_missing("openmm")
    def test_solvate_topology(self):
        ligand = Molecule.from_smiles("C1CN2C(=N1)SSC2=S")
        ligand.generate_conformers(n_conformers=1)

        solvated_topology = solvate_topology(
            ligand.to_topology(),
        )

        assert solvated_topology.molecule(0).to_smiles() == ligand.to_smiles()

        assert solvated_topology.molecule(1).to_smiles(explicit_hydrogens=False) == "O"

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
            assert solvated_topology.box_vectors[position[0], position[1]].m == [0.0]

        assert solvated_topology.box_vectors[0, 0] == solvated_topology.box_vectors[1, 1]
        assert solvated_topology.box_vectors[2, 0] == solvated_topology.box_vectors[2, 1]

    def test_packmol_add_negative_solvent_mass(self):
        ligand = MoleculeWithConformer.from_smiles("C1CN2C(=N1)SSC2=S")

        with pytest.raises(
            PACKMOLValueError,
            match="Solute mass is greater than target mass",
        ):
            solvate_topology(
                ligand.to_topology(),
                target_density=Quantity(1e-6, "kilogram/meter**3"),
            )

        with pytest.raises(
            PACKMOLValueError,
            match="Solute mass is greater than target mass",
        ):
            solvate_topology_nonwater(
                ligand.to_topology(),
                solvent=Molecule.from_smiles("CCCCCCO"),
                target_density=Quantity(1e-6, "kilogram/meter**3"),
            )

    @pytest.mark.skipif(
        has_package("mdtraj") or has_package("openmm"),
        reason="Test requires that MDTraj **and** OpenMM are not installed",
    )
    def test_noninteger_serial_error(self):
        """See issue #794."""
        with pytest.raises(
            PACKMOLRuntimeError,
            match="could not be parsed by RDKit",
        ):
            pack_box(
                molecules=[Molecule.from_smiles("CCO")],
                number_of_copies=[11112],
                box_shape=UNIT_CUBE,
                tolerance=1.0 * unit.angstrom,
                target_density=0.1 * unit.grams / unit.milliliters,
            )

    @pytest.mark.slow
    @skip_if_missing("mdtraj")
    def test_noninteger_serial_fallback(self):
        """See issue #794."""
        pack_box(
            molecules=[Molecule.from_smiles("CCO")],
            number_of_copies=[11112],
            box_shape=UNIT_CUBE,
            tolerance=1.0 * unit.angstrom,
            target_density=0.1 * unit.grams / unit.milliliters,
        )

    @pytest.mark.slow
    @skip_if_missing("mdtraj")
    def test_load_100_000_atoms(self):
        pack_box(
            molecules=[Molecule.from_smiles("CCO")],
            number_of_copies=[11112],
            box_shape=UNIT_CUBE,
            tolerance=1.0 * unit.angstrom,
            target_density=0.1 * unit.grams / unit.milliliters,
        )

    @pytest.mark.parametrize("use_local_path", [False, True])
    def test_save_error_on_convergence_failure(self, use_local_path):
        with pytest.raises(
            PACKMOLRuntimeError,
            match="failed with error code 173",
        ):
            pack_box(
                molecules=[Molecule.from_smiles("CCO")],
                number_of_copies=[100],
                box_shape=UNIT_CUBE,
                target_density=1000 * unit.grams / unit.milliliters,
                working_directory="." if use_local_path else None,
            )

        if use_local_path:
            assert "STOP 173" in open("packmol_error.log").read()
        else:
            assert not pathlib.Path("packmol_error.log").is_file()
