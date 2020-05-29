import pytest
import sympy
from simtk import unit as simtk_unit
import pint

from system.utils import simtk_to_pint, compare_sympy_expr


u = pint.UnitRegistry()

def test_simtk_to_pint():
    """Test conversion from SimTK Quantity to pint Quantity."""
    simtk_quantity = 10.0 * simtk_unit.nanometer

    pint_quantity = simtk_to_pint(simtk_quantity)

    assert pint_quantity == 10.0 * u.nanometer

@pytest.mark.parametrize(
    'expr1,expr2,result',
    [
        ('x+1', 'x+1', True),
        ('x+1', 'x**2+1', True),
        ('0.5*k*(th-th0)**2', 'k*(th-th0)**2', False),
        ('k*(1+cos(n*th-th0))', 'k*(1+cos(n*th-th0))', True),
    ]
)
def test_compare_sympy_expr(expr1, expr2, result):
    compare_sympy_expr(expr1, expr2) == result
