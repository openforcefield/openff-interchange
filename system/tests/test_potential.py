from sympy import Expr
from pint import Quantity

from system.potential import Potential


def test_potential():
    pot = Potential(
        name="X",
        expression=Expr("x+1"),
        independent_variables={Expr("j")},
        parameters={Expr("x"): Quantity(1)},
    )
