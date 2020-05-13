from typing import Union, Dict, Set

from pint import Quantity
from pydantic import BaseModel, validator
from sympy import Expr


class AnalyticalPotential(BaseModel):
    """Generic representation of an interaction potential."""
    name: str
    expression: Union[Expr, str]
    parameters: Dict[Union[Expr, str], Quantity]
    independent_variables: Set[Union[Expr, str]]

    @validator("expression")
    def is_valid_sympy_expr(cls, val):
        if isinstance(val, Expr):
            return val
        elif isinstance(val, str):
            return Expr(val)

    class Config:
        arbitrary_types_allowed = True
