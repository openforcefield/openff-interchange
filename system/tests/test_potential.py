import sympy
import pint

from system.potential import AnalyticalPotential, ParametrizedAnalyticalPotential
from system.utils import compare_sympy_expr
from system.tests.base_test import BaseTest


u = pint.UnitRegistry()


class TestPotential(BaseTest):
    def test_analytical_potential_constructor(self):
        pot = AnalyticalPotential(
            name="TestPotential",
            expression=sympy.sympify("m*x+b"),
            independent_variables={sympy.sympify("x")},
        )

        assert pot.name == "TestPotential"
        assert compare_sympy_expr(pot.expression, "m*x+b")

        pot_from_str = AnalyticalPotential(
            name="TestPotentialFromString",
            expression="m*x+b",
            independent_variables={"x"},
        )

        assert compare_sympy_expr(pot.expression, pot_from_str.expression)

    def test_parametrized_analytical_potential_constructor(self):
        pot = ParametrizedAnalyticalPotential(
            name="TestPotential",
            expression="m*x+b",
            independent_variables={"x"},
            parameters={"m": 0.5 * u.dimensionless, "b": -1.0 * u.dimensionless},
        )

        assert pot.name == "TestPotential"
        assert compare_sympy_expr(pot.expression, "m*x+b")
        assert "m" in pot.parameters.keys()
        assert "b" in pot.parameters.keys()
        assert pot.parameters["m"] == 0.5 * u.dimensionless
        assert pot.parameters["b"] == -1.0 * u.dimensionless
