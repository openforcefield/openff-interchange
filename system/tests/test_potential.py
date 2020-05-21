from sympy import Expr
import pint

from system.potential import AnalyticalPotential, ParametrizedAnalyticalPotential
from system.tests.base_test import BaseTest


u = pint.UnitRegistry()

class TestPotential(BaseTest):
    def test_analytical_potential_constructor(self):
        pot = AnalyticalPotential(
            name="TestPotential",
            expression=Expr("mx+b"),
            independent_variables={Expr("x")},
        )

        assert pot.name == "TestPotential"
        assert pot.expression == Expr("mx+b")

        pot_from_str = AnalyticalPotential(
            name="TestPotentialFromString",
            expression="mx+b",
            independent_variables={Expr("x")},
        )

        assert pot.expression == pot_from_str.expression

    def test_parametrized_analytical_potential_constructor(self):
        pot = ParametrizedAnalyticalPotential(
            name="TestPotential",
            expression=Expr("mx+b"),
            independent_variables={Expr("x")},
            parameters={"a": 0.5 * u.dimensionless, "b": -1.0 * u.dimensionless},
        )

        assert pot.name == "TestPotential"
        assert pot.expression == Expr("mx+b")
        assert "a" in pot.parameters.keys()
        assert "b" in pot.parameters.keys()
        assert pot.parameters["a"] == 0.5 * u.dimensionless
        assert pot.parameters["b"] == -1.0 * u.dimensionless
