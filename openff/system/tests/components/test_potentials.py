from openff.system.components.potentials import PotentialHandler
from openff.system.tests import BaseTest


class TestPotentialHandlerSubclassing(BaseTest):
    def test_dummy_potential_handler(self):
        handler = PotentialHandler(
            type="foo",
            expression="m*x+b",
        )
        assert handler.type == "foo"
        assert handler.expression == "m*x+b"
