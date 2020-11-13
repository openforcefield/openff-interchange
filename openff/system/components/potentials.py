from typing import Dict, List, Optional, Set, Union

import jax.numpy as jnp
from pydantic import BaseModel, validator
from sympy import Expr

from openff.system import unit


class Potential(BaseModel):
    """Base class for storing applied parameters"""

    parameters: Dict[str, Optional[Union[unit.Quantity, List, int]]] = dict()

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
            return val.args[0]
        elif isinstance(val, str):
            return val

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def store_matches(self):
        raise NotImplementedError

    def store_potentials(self):
        raise NotImplementedError

    def get_force_field_parameters(self):
        params: list = list()
        for potential in self.potentials.values():
            row = [val.magnitude for val in potential.parameters.values()]
            params.append(row)

        return jnp.array(params)

    def get_system_parameters(self, p=None):
        if p is None:
            p = self.get_force_field_parameters()
        mapping = self.get_mapping()
        q: List = list()

        for idx, val in enumerate(self.slot_map.keys()):
            q.append(p[mapping[self.slot_map[val]]])

        return jnp.array(q)

    def get_mapping(self) -> Dict:
        mapping: Dict = dict()
        for idx, key in enumerate(self.potentials.keys()):
            for p in self.slot_map.values():
                if key == p:
                    mapping.update({key: idx})

        return mapping

    def parametrize(self, p=None):
        if p is None:
            p = self.get_force_field_parameters()

        return self.get_system_parameters(p=p)

    def parametrize_partial(self):
        from functools import partial

        return partial(
            self.parametrize,
            mapping=self.get_mapping(),
        )

    def get_param_matrix(self):
        from functools import partial

        import jax

        p = self.get_force_field_parameters()

        parametrize_partial = partial(
            self.parametrize,
        )

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])
