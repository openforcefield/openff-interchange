"""
Unit and regression test for the system package.
"""

# Import package, test suite, and other packages as needed
import system
import pytest
import sys

def test_system_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "system" in sys.modules
