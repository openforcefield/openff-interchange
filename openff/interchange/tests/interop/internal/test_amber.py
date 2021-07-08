import mdtraj as md
import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers import get_amber_energies, get_gromacs_energies
from openff.interchange.tests import BaseTest
from openff.interchange.tests.utils import needs_gmx

kj_mol = unit.kilojoule / unit.mol


class TestAmber(BaseTest):
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

    @skip_if_missing("intermol")
    @needs_gmx
    @pytest.mark.slow()
    def test_amber_energy(self):
        """Basic test to see if the amber energy driver is functional"""
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        parsley = ForceField("openff_unconstrained-1.0.0.offxml")
        off_sys = Interchange.from_smirnoff(parsley, top)

        off_sys.box = [4, 4, 4]
        off_sys.positions = mol.conformers[0]

        omm_energies = get_gromacs_energies(off_sys, mdp="cutoff_hbonds")
        amb_energies = get_amber_energies(off_sys)

        omm_energies.compare(
            amb_energies,
            custom_tolerances={
                "Bond": 3.6 * kj_mol,
                "Angle": 0.2 * kj_mol,
                "Torsion": 1.9 * kj_mol,
                "vdW": 1.5 * kj_mol,
                "Electrostatics": 36.5 * kj_mol,
            },
        )
