from typing import Dict, Set, Union

from pydantic import BaseModel, validator
import sympy
from sympy import Expr

from . import unit


class AnalyticalPotential(BaseModel):
    """
    Generic representation of an interaction potential having an analytic
    form and lacking parameters.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    name : str, optional
        A string identifier
    smirks : str, optional
        The SMIRKS pattern associated with the potential,
        if using the SMIRNOFF specification
    expression : str or sympy.Expr
        A symbolic representation of the analytical expression defining the
        functional form of the potential
    independent_variables : set of str or sympy.Expr
        A symbolic representation of the independent variable of this potential
    """

    name: str = None
    smirks: str = None
    expression: Union[Expr, str]
    independent_variables: Union[str, Set[Union[Expr, str]]]

    @validator("expression")
    def is_valid_sympy_expr(cls, val):
        if isinstance(val, Expr):
            return str(val)
        elif isinstance(val, str):
            return val

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class ParametrizedAnalyticalPotential(AnalyticalPotential):
    """AnalyticalPotential but filled with parameters.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    name : str, optional
        A string identifier
    smirks : str, optional
        The SMIRKS pattern associated with the potential,
        if using the SMIRNOFF specification
    expression : str or sympy.Expr
        A symbolic representation of the analytical expression defining the
        functional form of the potential
    independent_variables : set of str or sympy.Expr
        A symbolic representation of the independent variable of this potential
    parameters : dict of [str or sympy.Expr : unit.Quantity]
    """
    parameters: Dict[Union[Expr, str], unit.Quantity]

    @validator("parameters")
    def is_valid(cls, val, values):

        symbols_in_expr = sympy.sympify(values['expression']).free_symbols
        symbols_in_indep_vars = sympy.symbols(values['independent_variables'])
        symbols_in_parameters = sympy.symbols(set(val.keys()))

        assert symbols_in_expr - symbols_in_indep_vars - symbols_in_parameters == set()

        return val
