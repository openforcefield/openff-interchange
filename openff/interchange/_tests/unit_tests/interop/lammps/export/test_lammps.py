from pathlib import Path

import lammps
import numpy
import pytest
from openff.toolkit import ForceField, Quantity, Topology, unit
from openff.utilities import temporary_cd

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_lmp
from openff.interchange.drivers import get_lammps_energies, get_openmm_energies

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
        """
        mol = MoleculeWithConformer.from_smiles(mol)
        mol.conformers[0] -= numpy.min(mol.conformers[0], axis=0)
        top = Topology.from_molecules(n_mols * [mol])

        box = numpy.zeros((3,3), dtype=float) * unit.angstrom

        box[0] = [51.34903463831951, 0, 0] * unit.angstrom
        box[1] = [-0.03849979989403723, 50.9134404144338, 0] * unit.angstrom
        box[2] = [-2.5907782992729538, 0.3720740833800747, 49.80705567557188] * unit.angstrom

        top.box_vectors = box

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
                "Nonbonded": 1 * unit.kilojoule_per_mole,
                "Torsion": 0.02 * unit.kilojoule_per_mole,
            },
        )

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
                "Nonbonded": 1 * unit.kilojoule_per_mole,
                "Torsion": 0.02 * unit.kilojoule_per_mole,
            },
        )

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
