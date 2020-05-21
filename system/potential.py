from typing import Dict, Set, Union

from pint import Quantity
from pydantic import BaseModel, validator
from sympy import Expr


class AnalyticalPotential(BaseModel):
    """
    Generic representation of an interaction potential having an analytic
    form and lacking parameters.
    """

    name: str
    expression: Union[Expr, str]
    independent_variables: Set[Union[Expr, str]]

    @validator("expression")
    def is_valid_sympy_expr(cls, val):
        if isinstance(val, Expr):
            return val
        elif isinstance(val, str):
            return Expr(val)

    class Config:
        arbitrary_types_allowed = True


class ParametrizedAnalyticalPotential(AnalyticalPotential):
    """AnalyticalPotential but filled with parameters."""

    parameters: Dict[Union[Expr, str], Quantity]

    @validator("parameters")
    def is_valid(cls, val, values):
        return val
