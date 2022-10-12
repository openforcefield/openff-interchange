import numpy as np
import parmed
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.constants import kj_mol
from openff.interchange.drivers import get_amber_energies, get_openmm_energies
from openff.interchange.tests import _BaseTest


class TestAmber(_BaseTest):
    @pytest.mark.slow()
    def test_inpcrd(self, sage):
        mol = Molecule.from_smiles(10 * "C")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(force_field=sage, topology=mol.to_topology())
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.positions = unit.nanometer * np.round(out.positions.m_as(unit.nanometer), 5)

        out.to_inpcrd("internal.inpcrd")
        out._to_parmed().save("parmed.inpcrd")

        coords1 = parmed.load_file("internal.inpcrd").coordinates
        coords2 = parmed.load_file("parmed.inpcrd").coordinates

        np.testing.assert_equal(coords1, coords2)

    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "smiles",
        [
            "C",
            "CC",  # Adds a proper torsion term(s)
            "C=O",  # Simplest molecule with any improper torsion
            "OC=O",  # Simplest molecule with a multi-term torsion
            "CCOC",  # This hits t86, which has a non-1.0 idivf
            "C1COC(=O)O1",  # This adds an improper, i2
        ],
    )
    def test_amber_energy(self, sage_unconstrained, smiles):
        """
        Basic test to see if the amber energy driver is functional.

        Note this test can only use the unconstrained version of Sage because sander applies SHAKE
        constraints in the single-point energy calculation, i.e. uses geometries with constraints
        applied, NOT what is in the coordinate file. See issue
        https://github.com/openforcefield/openff-interchange/issues/323
        """
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()

        off_sys = Interchange.from_smirnoff(sage_unconstrained, top)

        off_sys.box = [4, 4, 4]
        off_sys.positions = mol.conformers[0]

        omm_energies = get_openmm_energies(off_sys)
        amb_energies = get_amber_energies(off_sys)

        omm_energies.compare(
            amb_energies,
            custom_tolerances={
                "vdW": 0.018 * kj_mol,
                "Electrostatics": 0.01 * kj_mol,
            },
        )


class TestAmberResidues(_BaseTest):
    def test_single_residue_system(self, sage, ethanol):
        """
        Ensure a single-molecule system without specified residues writes something that ParmEd can read.

        See https://github.com/openforcefield/openff-interchange/pull/538#issue-1404910624
        """
        Interchange.from_smirnoff(sage, [ethanol]).to_prmtop("molecule.prmtop")

        # Use ParmEd as a soft validator of this file format
        parmed.amber.AmberParm("molecule.prmtop")
