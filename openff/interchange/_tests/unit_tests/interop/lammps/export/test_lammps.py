from pathlib import Path

import lammps
import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Topology, unit

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_lmp
from openff.interchange.components.mdconfig import MDConfig
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

    @pytest.mark.parametrize("sidelen", [2, 3])
    @pytest.mark.parametrize(
        "smiles",
        [ 
            "O",   # nothing particularly special about these molecules
            "CCO", # just testing that unique IDs hold for diverse chemistries
            "N1CCCC1",
            "c1cccc1c",

        ]
    )
    def test_unique_lammps_mol_ids(
        self,
        smiles: str,
        sage_unconstrained: ForceField,
        sidelen: int = 2,
    ) -> bool:
        """Test to see if interop.lammps.export._write_atoms() writes unique ids for each distinct Molecule"""
        assert isinstance(sidelen, int) and sidelen > 1

        # NOTE: the input file name can have an arbitrary name, but the data file !MUST! be named "out.lmp", as this is
        # the filename hard-coded into the read_data output of MDConfig.write_lammps_input()
        # Would be really nice to have more control over this programmatically in the future (perhaps have optional "data_file" kwarg?)
        cwd = Path.cwd()
        lmp_file_name: str = "temp"
        lammps_input_path = cwd / f"{lmp_file_name}.in"
        lammps_data_path = cwd / "out.lmp"

        # BUILD TOPOLOGY
        ## 1) compute effective radius as the greatest atomic distance from barycenter (avoids collisions when tiling)
        pilot_mol = MoleculeWithConformer.from_smiles(
            smiles,
        )  # this will serve as a prototype for all other Molecule copies in the Topology

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
            ],
        )

        ## 3) build topology by tiling
        tiled_top = Topology()
        for int_offset in xyz_offsets:
            mol = Molecule.from_smiles(smiles)
            mol.add_conformer(
                (conf_centered + 2 * r_eff * int_offset).to("nm"),
            )  # space copies by effective diameter
            tiled_top.add_molecule(mol)

        ## 3a) set periodic box tightly around extremem positions in filled topology
        box_dims = tiled_top.get_positions().ptp(axis=0)
        box_vectors = (
            numpy.eye(3) * box_dims
        )  # convert to diagonal matrix to get proper shape

        # EXPORT TO LAMMPS
        interchange = Interchange.from_smirnoff(
            sage_unconstrained,
            tiled_top,
        )
        interchange.box = box_vectors
        interchange.to_lammps(lammps_data_path)

        mdconfig = MDConfig.from_interchange(interchange)
        mdconfig.write_lammps_input(
            interchange=interchange, input_file=lammps_input_path
        )

        ## 4) EXTRACT ATOM INFO FROM WRITTEN LAMMPS FILE TO TEST IF MOLEUCLE IDS ARE BEING WRITTEN CORRECTLY
        with lammps.lammps(
            cmdargs=["-screen", "none", "-log", "none"]
        ) as lmp:  # Ask LAMMPS nicely not to spam console or produce stray log files
            lmp.file(
                "temp.in"
            )  # can't use lmp.command('read_data ...'), as atom/bond/pair/dihedral styles are not set in the Interchange-generated LAMMPS data file
            written_mol_ids = {
                mol_id
                for _, mol_id in zip(
                    range(lmp.get_natoms()), lmp.extract_atom("molecule")
                )
            }  # need to zip with range capped at n_atoms, otherwise will iterate forever

        # dispose of the lammps files once we've read them to leave no trace on disc
        lammps_input_path.unlink()
        lammps_data_path.unlink()

        # if all has gone well, we'd expect for each of the N molecules to have ids from [1...N]
        expected_mol_ids = {i + 1 for i in range(interchange.topology.n_molecules)}

        assert expected_mol_ids == written_mol_ids
