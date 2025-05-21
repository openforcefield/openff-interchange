import pytest

from openff.interchange._tests import get_test_file_path
from openff.interchange.exceptions import GROMACSParseError
from openff.interchange.interop.gromacs._import._import import from_files
from openff.interchange.interop.gromacs.models.models import RyckaertBellemansDihedral


def test_load_rb_torsions(monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    system = from_files(
        get_test_file_path("ethanol_rb_torsions.top"),
        get_test_file_path("ethanol_rb_torsions.gro"),
    )

    assert len(system.molecule_types["Compound"].dihedrals) > 0

    for torsion in system.molecule_types["Compound"].dihedrals:
        assert isinstance(torsion, RyckaertBellemansDihedral)


def test_parse_bad_characters(monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    with pytest.raises(
        GROMACSParseError,
        match="garbage.*FLAG 8EE0CCA23E",
    ):
        # gro file doesn't need to have contents, crash should happen on the
        # topology file which happens to be parsed first

        from_files(
            get_test_file_path("invalid.top"),
            get_test_file_path("invalid.gro"),
        )


def test_parse_invalid_directive_partway_through(monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    with pytest.raises(
        GROMACSParseError,
        match="directive.*flag",
    ):
        from_files(
            get_test_file_path("invalid_partway_through.top"),
            get_test_file_path("invalid.gro"),
        )
