from openff.interchange.components.potentials import PotentialHandler
from openff.interchange.tests import BaseTest


class TestPotentialHandlerSubclassing(BaseTest):
    def test_dummy_potential_handler(self):
        handler = PotentialHandler(
            type="foo",
            expression="m*x+b",
        )
        assert handler.type == "foo"
        assert handler.expression == "m*x+b"
