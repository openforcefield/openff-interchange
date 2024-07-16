import pytest

from openff.interchange import Interchange
from openff.interchange.warnings import InterchangeDeprecationWarning


class TestDeprecation:
    def test_potential_handler_deprecation(self):
        from openff.interchange.components.potentials import Collection

        with pytest.warns(
            InterchangeDeprecationWarning,
            match="`PotentialHandler` has been renamed to `Collection`.",
        ):
            from openff.interchange.components.potentials import PotentialHandler

        assert PotentialHandler is Collection

    @pytest.fixture
    def prepared_system(self, sage, water):
        return Interchange.from_smirnoff(sage, [water])

    def test_slot_map_deprecation(self, prepared_system):
        with pytest.warns(
            InterchangeDeprecationWarning,
            match="The `slot_map` attribute is deprecated. Use `key_map` instead.",
        ):
            prepared_system["vdW"].slot_map

    def test_handlers_deprecation(self, prepared_system):
        with pytest.warns(
            InterchangeDeprecationWarning,
            match="The `handlers` attribute is deprecated. Use `collections` instead.",
        ):
            prepared_system.handlers

    def test_plus_operator_warning(self, monkeypatch, prepared_system):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        with pytest.warns(
            InterchangeDeprecationWarning,
            match="combine.*instead",
        ):
            prepared_system + prepared_system
