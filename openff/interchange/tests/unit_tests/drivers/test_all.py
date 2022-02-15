"""
Test the behavior of the drivers.all module
"""
from distutils.spawn import find_executable

import pytest

from openff.interchange.drivers.all import get_all_energies
from openff.interchange.tests import _BaseTest


@pytest.mark.slow()
class TestDriversAll(_BaseTest):
    def test_skipping_drivers(self, ethanol_top, parsley):
        from openff.toolkit.topology import Molecule

        from openff.interchange import Interchange

        molecule = Molecule.from_smiles("C")
        molecule.generate_conformers(n_conformers=1)
        molecule.name = "MOL"
        topology = molecule.to_topology()

        out = Interchange.from_smirnoff(parsley, topology)
        out.positions = molecule.conformers[0]
        out.box = [4, 4, 4]

        summary = get_all_energies(out)

        assert ("GROMACS" in summary) == (find_executable("gmx") is not None)

        assert ("Amber" in summary) == (find_executable("sander") is not None)

        assert ("LAMMPS" in summary) == (find_executable("lmp_serial") is not None)

        number_expected_drivers = 1 + sum(
            int(find_executable(exec) is not None)
            for exec in ["gmx", "lmp_serial", "sander"]
        )

        assert len(summary) == number_expected_drivers
