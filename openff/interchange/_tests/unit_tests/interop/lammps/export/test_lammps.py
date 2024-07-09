from pathlib import Path
from typing import Any, Dict, Type

import numpy
import pytest
from openff.toolkit import Molecule, Topology, unit
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_lmp
from openff.interchange.drivers import get_lammps_energies, get_openmm_energies


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
    def test_to_lammps_single_mols(
        self, mol: str, sage_unconstrained: ForceField, n_mols: int
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

    def test_unique_lammps_mol_ids(
        self, smiles: str, sage_unconstrained: ForceField, sidelen: int = 0
    ) -> bool:
        """Test to see if interop.lammps.export._write_atoms() writes unique ids for each distinct Molecule"""
        temp_lammps_path = (
            Path.cwd() / "temp.lmp"
        )  # NOTE: lammps writer doesn't currently support IO-like object passing, so need to intercept output with temporary file

        # BUILD TOPOLOGY
        ## 0) generate pilot Molecule and conformer
        pilot_mol = Molecule.from_smiles(smiles)
        pilot_mol.generate_conformers(n_conformers=1)
        # pilot_mol.assign_partial_charges(partial_charge_method='gasteiger') # generate dummy charges for testing

        ## 1) compute effective radius as the greatest atomic distance from barycenter (avoids collisions when tiling)
        conf = pilot_mol.conformers[0]
        COM = conf.mean(axis=0)
        conf_centered = conf - COM

        radii = numpy.linalg.norm(conf_centered, axis=1)
        r_eff = radii.max()

        ## 2) generate 3D integer lattice to tile mols onto
        xyz_offsets = numpy.column_stack(
            [  # integral offsets from 0...(sidelen - 1) along 3 axes
                axis_offsets.ravel()
                for axis_offsets in numpy.meshgrid(
                    *[
                        numpy.arange(sidelen) for _ in range(3)
                    ],  # the 3 here is for 3-dimensions
                )
            ]
        )

        ## 3) build topology by tiling
        tiled_top = Topology()
        for int_offset in xyz_offsets:
            mol = Molecule.from_smiles(smiles)
            mol.add_conformer(
                (conf_centered + 2 * r_eff * int_offset).to("nm")
            )  # space copied by effective diameter
            tiled_top.add_molecule(mol)

        ## 3a) set periodic box tightly around extremem positions in filled topology
        box_dims = tiled_top.get_positions().ptp(axis=0)
        box_vectors = (
            numpy.eye(3) * box_dims
        )  # convert to diagonal matrix to get proper shape

        # EXPORT TO LAMMPS
        interchange = Interchange.from_smirnoff(
            sage_unconstrained, tiled_top
        )  # , charge_from_molecules=[pilot_mol])
        # UNRELATED QUESTION: why are waters not correctly recognized?
        # PDB residue name on output is always UNK for Molecule.from_smiles("O"), EVEN when using tip3p.offxml)
        interchange.box = box_vectors
        interchange.to_lammps(temp_lammps_path)

        # EXTRACT ATOM INFO FROM WRITTEN LAMMPS FILE TO TEST IF MOLEUCLE IDS ARE BEING WRITTEN CORRECTLY
        with temp_lammps_path.open(
            "r"
        ) as lmp_file:  # pull out text from temporary lammps file ...
            all_lines = [
                line for line in lmp_file.read().split("\n") if line
            ]  # ... separate by newlines, remove empty lines ...
            atom_lines = all_lines[
                all_lines.index("Atoms") + 1 : all_lines.index("Bonds")
            ]  # ... extract atoms block ...
        temp_lammps_path.unlink()  # ...and finally, unceremoniously kill the file once we're done with it

        ## SIDENOTE: would be nice if there was an easier, string-parsing-free way to extract this info
        ## Could enable some kind of Interchange.from_lammps() functionality in the future, perhaps
        def extract_info_from_lammps_atom_lines(atom_line: str) -> dict[str, Any]:
            """Parse atom info from a single atom-block linds in a .lmp/.lammps files"""
            KEYWORDS: dict[str, type] = {  # reference for the distinct
                "atom_index": int,
                "molecule_index": int,
                "atom_type": int,
                "charge": float,
                "x-pos": float,
                "y-pos": float,
                "z-pos": float,
            }

            return {
                field_label: FieldType(str_val)
                for str_val, (field_label, FieldType) in zip(
                    atom_line.split("\t"), KEYWORDS.items()
                )
            }

        written_mol_ids: set[int] = set()
        for atom_line in atom_lines:
            atom_info = extract_info_from_lammps_atom_lines(atom_line)
            written_mol_ids.add(atom_info["molecule_index"])
        expected_mol_ids = {
            i + 1 for i in range(interchange.topology.n_molecules)
        }  # we'd like for each of the N molecules to have ids from [1...N]

        return expected_mol_ids == written_mol_ids
