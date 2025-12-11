from packaging.version import Version

from openff.interchange import __version__


def test_version_fallback():
    assert Version(__version__) > Version("0.0.0")
