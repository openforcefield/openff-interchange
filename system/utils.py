from evaluator.utils.openmm import openmm_quantity_to_pint                                                                      
import sympy


def pint_to_openmm(quantity):
    """Convert a pint Quantity to an OpenMM unit."""
    raise NotImplementedError()

def simtk_to_pint(simtk_quantity):
    """Convert an OpenMM unit to a pint Quantity."""
    # TODO: Implement copy of this function or host in a shared location
    return openmm_quantity_to_pint(smitk_quantity)

def compare_sympy_expr(expr1, expr2):
    """Checks if two expression-likes are equivalent."""
    expr1 = sympy.sympify(expr1)
    expr2 = sympy.sympify(expr2)

    return sympy.simplify(expr1 - expr2) == 0
