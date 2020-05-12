from typing import Dict, Set

from pint import Quantity
from pydantic import BaseModel, validator
from sympy import Expr


class Potential(BaseModel):
    """Generic representation of an interaction potential."""
    name: str
    expression: Expr
    parameters: Dict[Expr, Quantity]
    independent_variables: Set[Expr]

    @validator("expression")
    def is_valid_sympy_expr(cls, val):
        if isinstance(val, Expr):
            return val
        elif isinstance(val, str):
            return Expr(val)

    class Config:
        arbitrary_types_allowed = True
