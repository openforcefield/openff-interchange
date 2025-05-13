import packaging.version
import pytest

from openff.interchange.interop.lammps.export.provenance import get_lammps_version


def test_get_lammps_version():
    pytest.importorskip("lammps")

    assert get_lammps_version() > packaging.version.Version("2020")

    assert get_lammps_version() < packaging.version.Version("2100")
