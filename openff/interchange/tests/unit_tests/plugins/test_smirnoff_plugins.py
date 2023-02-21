import pytest
from nonbonded_plugins import (
    BuckinghamHandler,
    DoubleExponentialHandler,
    SMIRNOFFBuckinghamCollection,
    SMIRNOFFDoubleExponentialCollection,
)
from openff.toolkit.typing.engines.smirnoff.plugins import load_handler_plugins

from openff.interchange.plugins import load_smirnoff_plugins


def test_load_handler_plugins():
    # This does not test Interchange per se but this behavior is necessary for the other tests to function
    available_plugins = load_handler_plugins()

    assert BuckinghamHandler in available_plugins
    assert DoubleExponentialHandler in available_plugins


def test_load_smirnoff_plugins():
    available_plugins = load_smirnoff_plugins()

    assert SMIRNOFFBuckinghamCollection in available_plugins
    assert SMIRNOFFDoubleExponentialCollection in available_plugins


@pytest.mark.importorskip("deforcefields")
class TestDoubleExponential:
    def test_loadable(self):
        from openff.toolkit import ForceField

        ForceField("de-force-1.0.0.offxml", load_plugins=True)
