import sympy
import pint


def pint_to_simtk(quantity):
    """Convert a pint Quantity to an OpenMM unit."""
    raise NotImplementedError()


def simtk_to_pint(simtk_quantity):
    """
    Convert a SimTK Quantity (OpenMM Quantity) to a pint Quantity.

    Note: This function is adapted from evaluator.utils.openmm.openmm_quantity_to_pint,
    as part of the OpenFF Evaluator, Copyright (c) 2019 Open Force Field Consortium.
    """
    simtk_unit = simtk_quantity.unit
    simtk_value = simtk_quantity.value_in_unit(simtk_unit)

    u = pint.UnitRegistry()
    pint_unit = u(simtk_unit.get_symbol())
    pint_quantity = simtk_value * pint_unit

    return pint_quantity


def compare_sympy_expr(expr1, expr2):
    """Checks if two expression-likes are equivalent."""
    expr1 = sympy.sympify(expr1)
    expr2 = sympy.sympify(expr2)

    return sympy.simplify(expr1 - expr2) == 0
