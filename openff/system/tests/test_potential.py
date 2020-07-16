import sympy
from pydantic import parse_obj_as

from .. import unit
from ..potential import AnalyticalPotential, ParametrizedAnalyticalPotential
from ..tests.base_test import BaseTest
from ..utils import compare_sympy_expr


class TestPotential(BaseTest):
    def test_analytical_potential_constructor(self):
        pot = AnalyticalPotential(
            name="TestPotential",
            smirks="[#6]",
            expression=sympy.sympify("m*x+b"),
            independent_variables={sympy.sympify("x")},
        )

        assert pot.name == "TestPotential"
        assert pot.smirks == "[#6]"
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
            parameters={"m": 0.5 * unit.dimensionless, "b": -1.0 * unit.dimensionless},
        )

        assert pot.name == "TestPotential"
        assert compare_sympy_expr(pot.expression, "m*x+b")
        assert "m" in pot.parameters.keys()
        assert "b" in pot.parameters.keys()
        assert pot.parameters["m"] == 0.5 * unit.dimensionless
        assert pot.parameters["b"] == -1.0 * unit.dimensionless

    def test_serialization(self):
        pot = AnalyticalPotential(
            smirks="[#6]", expression="m*x+b", independent_variables="x",
        )

        pot_param = ParametrizedAnalyticalPotential(
            smirks=pot.smirks,
            expression="m*x+b",
            independent_variables={"x"},
            parameters={"m": 1 * unit.dimensionless, "b": 1 * unit.dimensionless},
        )

        # Depending on the equality operator may be dangerous
        pot == parse_obj_as(AnalyticalPotential, pot.dict())
        pot == parse_obj_as(ParametrizedAnalyticalPotential, pot_param.dict())
