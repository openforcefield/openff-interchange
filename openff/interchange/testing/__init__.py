import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange.testing.utils import _top_from_smiles
from openff.interchange.utils import get_test_file_path


class _BaseTest:
    @pytest.fixture(autouse=True)
    def _initdir(self, tmpdir):
        tmpdir.chdir()

    # TODO: group fixtures up as dicts, i.e. argon['forcefield'], argon['topology'], ...
    @pytest.fixture()
    def argon_ff(self):
        """Fixture that loads an SMIRNOFF XML for argon."""
        return ForceField(get_test_file_path("argon.offxml"))

    @pytest.fixture()
    def argon_top(self):
        """Fixture that builds a simple arogon topology."""
        return _top_from_smiles("[#18]")

    @pytest.fixture()
    def ethanol_top(self):
        """Fixture that builds a simple four ethanol topology."""
        return _top_from_smiles("CCO", n_molecules=4)

    @pytest.fixture()
    def parsley(self):
        return ForceField("openff-1.0.0.offxml")

    @pytest.fixture()
    def parsley_unconstrained(self):
        return ForceField("openff_unconstrained-1.0.0.offxml")
