import mdtraj as md
import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers import get_amber_energies, get_openmm_energies
from openff.interchange.testing import _BaseTest

kj_mol = unit.kilojoule / unit.mol


class TestAmber(_BaseTest):
    def test_inpcrd(self, parsley):
        mol = Molecule.from_smiles(10 * "C")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(force_field=parsley, topology=mol.to_topology())
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.positions = unit.nanometer * np.round(out.positions.m_as(unit.nanometer), 5)

        out.to_inpcrd("internal.inpcrd")
        out._to_parmed().save("parmed.inpcrd")

        coords1 = pmd.load_file("internal.inpcrd").coordinates
        coords2 = pmd.load_file("parmed.inpcrd").coordinates

        np.testing.assert_equal(coords1, coords2)

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
    def test_amber_energy(self, smiles):
        """Basic test to see if the amber energy driver is functional"""
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        parsley = ForceField("openff_unconstrained-2.0.0.offxml")
        off_sys = Interchange.from_smirnoff(parsley, top)

        off_sys.box = [4, 4, 4]
        off_sys.positions = mol.conformers[0]

        omm_energies = get_openmm_energies(off_sys)
        amb_energies = get_amber_energies(off_sys)

        omm_energies.compare(
            amb_energies,
            custom_tolerances={
                "vdW": 0.02 * kj_mol,
                "Electrostatics": 0.05 * kj_mol,
            },
        )
