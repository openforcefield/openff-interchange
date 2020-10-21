from typing import Dict, Optional, Set, Union

from pydantic import BaseModel, validator
from sympy import Expr

from openff.system import unit


class Potential(BaseModel):
    """Base class for storing applied parameters"""

    parameters: Dict[str, Optional[unit.Quantity]] = dict()

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class PotentialHandler(BaseModel):
    """Base class for storing parametrized force field data"""

    name: str
    expression: Union[Expr, str]
    independent_variables: Union[str, Set[Union[Expr, str]]]
    slot_map: Dict[tuple, str] = dict()
    potentials: Dict[str, Potential] = dict()

    @validator("expression")
    def is_valid_sympy_expr(cls, val):
        if isinstance(val, Expr):
            return str(val)
        elif isinstance(val, str):
            return val

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def store_matches(self):
        raise NotImplementedError

    def store_potentials(self):
        raise NotImplementedError
