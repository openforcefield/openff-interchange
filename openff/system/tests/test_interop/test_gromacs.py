import pytest
from openff.utilities.testing import skip_if_missing

from openff.system.components.system import System
from openff.system.exceptions import UnsupportedExportError
from openff.system.tests.base_test import BaseTest


class TestGROMACS(BaseTest):
    @skip_if_missing("parmed")
    def test_set_mixing_rule(self, ethanol_top, parsley):
        import parmed as pmd

        openff_sys = System.from_smirnoff(force_field=parsley, topology=ethanol_top)

        openff_sys.to_top("lorentz.top")
        top_file = pmd.load_file("lorentz.top")
        assert top_file.combining_rule == "lorentz"

        openff_sys["vdW"].mixing_rule = "geometric"

        openff_sys.to_top("geometric.top")
        import os

        os.system("cat geometric.top")
        top_file = pmd.load_file("geometric.top")
        assert top_file.combining_rule == "geometric"

    @pytest.mark.xfail(
        reason="cannot test unsupported mixing rules in GROMACS with current SMIRNOFFvdWHandler model"
    )
    def test_unsupported_mixing_rule(self, ethanol_top, parsley):
        # TODO: Update this test when the model supports more mixing rules than GROMACS does
        openff_sys = System.from_smirnoff(force_field=parsley, topology=ethanol_top)
        openff_sys["vdW"].mixing_rule = "kong"

        with pytest.raises(UnsupportedExportError, match="rule `geometric` not compat"):
            openff_sys.to_top("out.top")
