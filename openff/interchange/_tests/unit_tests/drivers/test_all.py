"""
Test the behavior of the drivers.all module
"""

import math

import pandas
import pytest
from openff.toolkit import Quantity
from openff.utilities.testing import skip_if_missing

from openff.interchange._tests import (
    HAS_GROMACS,
    HAS_LAMMPS,
    HAS_SANDER,
    MoleculeWithConformer,
    needs_gmx,
    needs_lmp,
    needs_not_gmx,
    needs_not_lmp,
    needs_not_sander,
    needs_sander,
)
from openff.interchange.drivers.all import get_all_energies, get_summary_data


@skip_if_missing("openmm")
class TestDriversAll:
    @pytest.fixture
    def basic_interchange(self, sage_unconstrained):
        molecule = MoleculeWithConformer.from_smiles("CCO")
        molecule.name = "MOL"
        topology = molecule.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        return sage_unconstrained.create_interchange(topology)

    @needs_gmx
    @needs_lmp
    @needs_sander
    def test_all_with_all(self, basic_interchange):
        summary = get_all_energies(basic_interchange)

        assert len(summary) == 4

    @needs_not_gmx
    @needs_not_lmp
    @needs_not_sander
    def test_all_with_minimum(self, basic_interchange):
        summary = get_all_energies(basic_interchange)

        assert len(summary) == 1

    def test_skipping(self, basic_interchange):
        summary = get_all_energies(basic_interchange)

        assert ("GROMACS" in summary) == HAS_GROMACS
        assert ("Amber" in summary) == HAS_SANDER
        assert ("LAMMPS" in summary) == HAS_LAMMPS

    # TODO: Also run all of this with h-bond constraints
    def test_summary_data(self, basic_interchange):
        summary = get_summary_data(basic_interchange)

        if len(summary) < 2:
            pytest.skip("Not enough engines available to compare results")

        assert isinstance(summary, pandas.DataFrame)

        assert "OpenMM" in summary.index

        assert ("GROMACS" in summary.index) == HAS_GROMACS
        assert ("Amber" in summary.index) == HAS_SANDER
        assert ("LAMMPS" in summary.index) == HAS_LAMMPS

        # Check that (some of) the data is reasonable, this tolerance should be greatly reduced
        # See https://github.com/openforcefield/openff-interchange/issues/632
        for key in ["Bond", "Angle", "Torsion"]:
            assert summary.describe().loc["std", key] < 0.001, f"{key} failed comparison"

        # Check that (some of) the data did not NaN out
        for val in summary["Torsion"].to_dict().values():
            assert not math.isnan(val)
