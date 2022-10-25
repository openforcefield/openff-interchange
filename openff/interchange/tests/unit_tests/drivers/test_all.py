"""
Test the behavior of the drivers.all module
"""
from shutil import which

import pytest

from openff.interchange.drivers.all import get_all_energies
from openff.interchange.drivers.gromacs import _find_gromacs_executable
from openff.interchange.tests import _BaseTest


@pytest.mark.slow()
class TestDriversAll(_BaseTest):
    def test_skipping_drivers(self, ethanol_top, sage):
        from openff.toolkit.topology import Molecule

        from openff.interchange import Interchange

        molecule = Molecule.from_smiles("CCO")
        molecule.generate_conformers(n_conformers=1)
        molecule.name = "MOL"
        topology = molecule.to_topology()

        out = Interchange.from_smirnoff(sage, topology)
        out.positions = molecule.conformers[0]
        out.box = [4, 4, 4]

        summary = get_all_energies(out)
        assert ("GROMACS" in summary) == (_find_gromacs_executable() is not None)

        assert ("Amber" in summary) == (which("sander") is not None)

        # FIXME: Add back when LAMMPS export fixed
        # assert ("LAMMPS" in summary) == (_find_lammps_executable() is not None)

        assert len(summary) == 3
