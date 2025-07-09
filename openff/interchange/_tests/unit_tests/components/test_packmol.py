"""Unit test(s) for old PACKMOL wrapper import path."""

import pytest

from openff.interchange.warnings import InterchangeDeprecationWarning


def test_deprecation_warning_thrown():
    with pytest.warns(
        InterchangeDeprecationWarning,
        match="openff.interchange.packmol` instead",
    ):
        from openff.interchange.components._packmol import UNIT_CUBE  # noqa: F401
