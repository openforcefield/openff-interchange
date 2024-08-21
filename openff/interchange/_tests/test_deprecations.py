import pytest

from openff.interchange import Interchange
from openff.interchange.warnings import InterchangeDeprecationWarning


class TestDeprecation:

    @pytest.fixture
    def prepared_system(self, sage, water):
        return Interchange.from_smirnoff(sage, [water])

    def test_plus_operator_warning(self, monkeypatch, prepared_system):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        with pytest.warns(
            InterchangeDeprecationWarning,
            match="combine.*instead",
        ):
            prepared_system + prepared_system
