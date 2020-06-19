from pkg_resources import resource_filename
import pathlib

import sympy
from simtk import openmm

from openforcefield.utils import unit_to_string

from . import unit
from .types import UnitArray


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

    target_unit = unit_to_string(simtk_unit)
    target_unit = unit.Unit(target_unit)

    return UnitArray(simtk_value, units=target_unit)


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


def get_nonbonded_force_from_openmm_system(omm_system):
    for force in omm_system.getForces():
        if type(force) == openmm.NonbondedForce:
            return force


def get_partial_charges_from_openmm_system(omm_system):
    """Get partial charges from an OpenMM system as a unit.Quantity array."""
    # TODO: deal with virtual sites
    n_particles = omm_system.getNumParticles()
    force = get_nonbonded_force_from_openmm_system(omm_system)
    # TODO: don't assume the partial charge will always be parameter 0
    partial_charges = [simtk_to_pint(force.getParticleParameters(idx)[0]) for idx in range(n_particles)]
    partial_charges = unit.Quantity.from_list(partial_charges)
    return partial_charges


try:
    import jax
    jax_available = True
except ImportError:
    jax_available = False
