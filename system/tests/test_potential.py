from sympy import Expr
from pint import Quantity

from system.potential import AnalyticalPotential
from system.tests.base_test import BaseTest


class TestPotential(BaseTest):
    def test_analytical_potential_constructor(self):
        pot = AnalyticalPotential(
            name="X",
            expression=Expr("x+1"),
            independent_variables={Expr("j")},
            parameters={Expr("x"): Quantity(1)},
        )

        assert pot.name == "X"
        assert pot.expression == Expr("x+1")
        assert pot.independent_variables == {Expr("j")}
        assert pot.parameters == dict({Expr("x"): Quantity(1)})

        pot_from_str = AnalyticalPotential(
            name="X",
            expression="x+1",
            independent_variables={Expr("j")},
            parameters={Expr("x"): Quantity(1)},
        )

        assert pot.expression == pot_from_str.expression
