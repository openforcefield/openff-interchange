import copy
from pathlib import Path

import numpy
import pytest
from openff.toolkit import ForceField, Quantity, Topology, unit
from openff.utilities import temporary_cd

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_lmp
from openff.interchange.constants import kj_mol
from openff.interchange.drivers import get_lammps_energies, get_openmm_energies
from openff.interchange.exceptions import EnergyError

rng = numpy.random.default_rng(821)

@needs_lmp
class TestLammps:
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
    def test_to_lammps_single_mols_triclinic(
        self,
        mol: str,
        sage_unconstrained: ForceField,
        n_mols: int,
    ) -> None:
        """
        Test that Interchange.to_openmm Interchange.to_lammps report sufficiently similar energies
        in triclinic simulation boxes.
        This test is only sensitive to nonbonded interactions. Due to inherent differences in how
        long-ranged interactions are handled in openmm/lammps, there will be a baseline deviation
        between the two codes that goes beyond this test. This test is designed to test whether the
        triclinic box representations give no error greater than the ones from the orthogonal box test.
        """
        mol = MoleculeWithConformer.from_smiles(mol)
        mol.conformers[0] -= numpy.min(mol.conformers[0], axis=0)
        top = Topology.from_molecules(n_mols * [mol])

<<<<<<< HEAD
        box_t = numpy.zeros((3,3), dtype=float) * unit.angstrom
=======
        box = numpy.zeros((3, 3), dtype=float) * unit.angstrom
>>>>>>> 93384ed44aad23798252ae06fc348d2d75b4857b

        box_t[0] = [51.34903463831951, 0, 0] * unit.angstrom
        box_t[1] = [-0.03849979989403723, 50.9134404144338, 0] * unit.angstrom
        box_t[2] = [-2.5907782992729538, 0.3720740833800747, 49.80705567557188] * unit.angstrom

        box_o = 50. * numpy.eye(3) * unit.angstrom

        top.box_vectors = box_o

        if n_mols == 1:
            positions = mol.conformers[0]
        elif n_mols == 2:
            positions = numpy.concatenate(
                [mol.conformers[0], mol.conformers[0] + 1.5 * unit.nanometer],
            )

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)
        interchange.positions = positions
        interchange.box = box_o

        reference_o = get_openmm_energies(
            interchange=interchange,
            round_positions=3,
        )

        lmp_energies_o = get_lammps_energies(
            interchange=interchange,
            round_positions=3,
        )

        interchange.box = box_t
        reference_t = get_openmm_energies(
            interchange=interchange,
            round_positions=3,
        )

        lmp_energies_t = get_lammps_energies(
            interchange=interchange,
            round_positions=3,
        )

        diff_o = reference_o.diff(lmp_energies_o)
        diff_t = reference_t.diff(lmp_energies_t)

        assert abs(diff_o["Nonbonded"] - diff_t["Nonbonded"]) < 0.1 * unit.kilojoule_per_mole


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
    def test_to_lammps_single_mols(
        self,
        mol: str,
        sage_unconstrained: ForceField,
        n_mols: int,
    ) -> None:
        """
        Test that Interchange.to_openmm Interchange.to_lammps report sufficiently similar energies.

        TODO: Tighten tolerances
        TODO: Test periodic and non-periodic
        """
        mol = MoleculeWithConformer.from_smiles(mol)
        mol.conformers[0] -= numpy.min(mol.conformers[0], axis=0)
        top = Topology.from_molecules(n_mols * [mol])

        top.box_vectors = 5.0 * numpy.eye(3) * unit.nanometer

        if n_mols == 1:
            positions = mol.conformers[0]
        elif n_mols == 2:
            positions = numpy.concatenate(
                [mol.conformers[0], mol.conformers[0] + 1.5 * unit.nanometer],
            )

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)
        interchange.positions = positions
        interchange.box = top.box_vectors

        reference = get_openmm_energies(
            interchange=interchange,
            round_positions=3,
        )

        lmp_energies = get_lammps_energies(
            interchange=interchange,
            round_positions=3,
        )

        lmp_energies.compare(
            reference,
            tolerances={
                "Nonbonded": 0.4 * unit.kilojoule_per_mole,
                "Torsion": 0.02 * unit.kilojoule_per_mole,
            },
        )

    @pytest.mark.parametrize(
        "n_mols",
        [1, 2],
    )
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
    def test_to_lammps_single_mols_relative_to_perturbed_coordinates(
        self,
        mol: str,
        sage_unconstrained: ForceField,
        n_mols: int,
    ) -> None:
        """
        Test that Interchange.to_openmm Interchange.to_lammps report sufficiently similar relative energies between
        their initial coordinates and coordinates that have been randomly perturbed.

        TODO: Tighten tolerances
        TODO: Test periodic and non-periodic
        """
        rng = numpy.random.default_rng(821)
        mol = MoleculeWithConformer.from_smiles(mol)
        mol.conformers[0] -= numpy.min(mol.conformers[0], axis=0)
        top = Topology.from_molecules(n_mols * [mol])

        top.box_vectors = 5.0 * numpy.eye(3) * unit.nanometer

        if n_mols == 1:
            positions = mol.conformers[0]
        elif n_mols == 2:
            positions = numpy.concatenate(
                [mol.conformers[0], mol.conformers[0] + 1.5 * unit.nanometer],
            )
        perturbed_positions = positions + rng.random(positions.shape) * 0.1 * unit.angstrom

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)
        interchange.positions = positions
        interchange.box = top.box_vectors

        perturbed_interchange = copy.deepcopy(interchange)
        perturbed_interchange.positions = perturbed_positions

        reference = get_openmm_energies(
            interchange=interchange,
            round_positions=3,
            combine_nonbonded_forces=False,
        )
        perturbed_reference = get_openmm_energies(
            interchange=perturbed_interchange,
            round_positions=3,
            combine_nonbonded_forces=False,
        )

        lmp_energies = get_lammps_energies(
            interchange=interchange,
            round_positions=3,
        )
        perturbed_lmp_energies = get_lammps_energies(
            interchange=perturbed_interchange,
            round_positions=3,
        )

        reference_energy_differences = {
            key: diff.to(kj_mol) for key, diff in reference.diff(perturbed_reference).items()
        }

        lmp_energy_differences = {
            key: diff.to(kj_mol) for key, diff in lmp_energies.diff(perturbed_lmp_energies).items()
        }

        tolerances = {
            "Bond": 1e-11 * kj_mol,
            "Angle": 1e-10 * kj_mol,
            "Torsion": 1e-12 * kj_mol,
            "vdW": 1e-7 * kj_mol,
            "Electrostatics": 1e-1 * kj_mol,
        }

        errors = dict()
        for key, lmp_diff in lmp_energy_differences.items():
            error = abs(lmp_diff - reference_energy_differences[key])
            if error > tolerances[key]:
                errors[key] = error

        if errors:
            raise EnergyError(errors)

    @pytest.mark.parametrize("n_mols", [2, 3])
    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",
            "N1CCCC1",
            "c1ccccc1",
        ],
    )
    def test_unique_lammps_mol_ids(
        self,
        smiles,
        sage_unconstrained,
        n_mols,
    ) -> bool:
        """Test to see if interop.lammps.export._write_atoms() writes unique ids for each distinct Molecule"""
        import lammps

        molecule = MoleculeWithConformer.from_smiles(smiles)
        topology = Topology.from_molecules(n_mols * [molecule])

        # Just use random positions since we're testing molecule IDs, not physics
        topology.set_positions(
            Quantity(
                rng.random((topology.n_atoms, 3)),
                "nanometer",
            ),
        )
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        with temporary_cd():
            lammps_prefix = Path.cwd() / "lammps_test"

            interchange = sage_unconstrained.create_interchange(topology)
            interchange.to_lammps(lammps_prefix)

            # Extract molecule IDs from data file
            with lammps.lammps(
                cmdargs=["-screen", "none", "-log", "none"],
            ) as lmp:
                lmp.file(
                    "lammps_test_pointenergy.in",
                )
                written_mol_ids = {
                    mol_id
                    for _, mol_id in zip(
                        range(lmp.get_natoms()),
                        lmp.extract_atom("molecule"),
                    )
                }

        # these are expected to be [1...N] for each of N molecules
        expected_mol_ids = {i + 1 for i in range(interchange.topology.n_molecules)}

        assert expected_mol_ids == written_mol_ids

    @pytest.mark.parametrize(
        "mol",
        [
            "C=O",  # Simplest molecule with any improper torsion
            "OC=O",  # Simplest molecule with a multi-term torsion
        ],
    )
    def test_to_lammps_with_type_labels(
        self,
        mol: str,
        sage_unconstrained: ForceField,
        tmp_path,
    ) -> None:
        import lammps

        from openff.interchange.exceptions import LAMMPSRunError

        mol = MoleculeWithConformer.from_smiles(mol)
        mol.conformers[0] -= numpy.min(mol.conformers[0], axis=0)
        top = Topology.from_molecules([mol])

        top.box_vectors = 5.0 * numpy.eye(3) * unit.nanometer
        positions = mol.conformers[0]

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)
        interchange.positions = positions
        interchange.box = top.box_vectors

        interchange.to_lammps(tmp_path / "out", include_type_labels=True)

        runner = lammps.lammps(cmdargs=["-screen", "none", "-nocite"])

        try:
            runner.file("out_pointenergy.in")
        except Exception as error:
            raise LAMMPSRunError from error
