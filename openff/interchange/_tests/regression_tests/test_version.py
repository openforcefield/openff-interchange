from openff.interchange import __version__

from packaging.version import Version

def test_version_fallback():
    assert Version(__version__) > Version("0.0.0")
