import shutil

import pytest
from openff.utilities import has_executable, has_package


@pytest.mark.skipif(
    not has_executable("packmol"),
    reason="Packmol is not available on this system",
)
def test_basic_reexport():
    """Test that the packmol module can be imported."""
    from openff.interchange.components._packmol import _find_packmol, pack_box

    assert _find_packmol() == shutil.which("packmol")

    assert "interchange" not in pack_box.__module__


@pytest.mark.skipif(
    has_package("openff.packmol"),
    reason="openff.packmol is not installed",
)
def test_import_error_when_openff_packmol_not_installed():
    with pytest.raises(
        ImportError,
    ):
        from openff.interchange.components._packmol import pack_box

        assert "interchange" not in pack_box.__module__
