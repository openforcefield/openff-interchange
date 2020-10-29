from sympy import Expr

from openff.system import unit
from openff.system.components.potentials import Potential, PotentialHandler

from ..tests.base_test import BaseTest


class TestPotentialHandler(BaseTest):
    """Test the behavior of PotentialHandler objects"""

    def test_potential_handler_constructor(self):
        """Test a basic construction"""
        PotentialHandler(name="foo", expression="m*x+b", independent_variables="x")

    def test_sympy_expr(self):
        """Test that sympy Expr are parsed and stored as str"""
        as_str = PotentialHandler(
            name="foo", expression="m*x+b", independent_variables="x"
        )
        as_expr = PotentialHandler(
            name="foo", expression=Expr("m*x+b"), independent_variables="x"
        )

        assert as_str.dict() == as_expr.dict()


class TestPotential(BaseTest):
    """Test the behavior of Potential objects"""

    def test_potential_constructor(self):
        """Test the constructor of a single Potential in a PotentialHandler"""
        handler = PotentialHandler(
            name="foo", expression="m*x+b", independent_variables="x"
        )
        handler.slot_map.update({(0,): "line1"})
        handler.potentials.update(
            {
                "line1": Potential(
                    parameters={"m": 1 * unit.meter, "b": -1 * unit.second}
                )
            }
        )

        param = handler.potentials[handler.slot_map[(0,)]]

        assert param.parameters["m"] == 1 * unit.meter
        assert param.parameters["b"] == -1 * unit.second
