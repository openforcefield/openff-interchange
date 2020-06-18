import sympy

from pkg_resources import resource_filename
import pathlib

from . import unit


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

    pint_unit = unit(simtk_unit.get_name())
    pint_quantity = simtk_value * pint_unit

    return pint_quantity


def compare_sympy_expr(expr1, expr2):
    """Checks if two expression-likes are equivalent."""
    expr1 = sympy.sympify(expr1)
    expr2 = sympy.sympify(expr2)

    return sympy.simplify(expr1 - expr2) == 0


def get_test_file_path(test_file):
    dir_path = resource_filename('system', 'tests/files/')
    test_file_path = pathlib.Path(dir_path).joinpath(test_file)

    if test_file_path.is_file():
        return test_file_path
    else:
        raise FileNotFoundError(
            f'could not file file {test_file} in path {dir_path}'
        )


try:
    import jax
    jax_available = True
except ImportError:
    jax_available = False
