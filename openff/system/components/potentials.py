from typing import TYPE_CHECKING, Dict, List, Set, Union

from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from pydantic import validator

from openff.system.exceptions import InvalidExpressionError
from openff.system.models import DefaultModel, PotentialKey, TopologyKey
from openff.system.types import ArrayQuantity, FloatQuantity
from openff.system.utils import requires_package

if TYPE_CHECKING:
    from openff.system.components.misc import OFFBioTop


class Potential(DefaultModel):
    """Base class for storing applied parameters"""

    # ... Dict[str, FloatQuantity] = dict()
    parameters: Dict = dict()

    @validator("parameters")
    def validate_parameters(cls, v):
        for key, val in v.items():
            if isinstance(val, list):
                v[key] = ArrayQuantity.validate_type(val)
            else:
                v[key] = FloatQuantity.validate_type(val)
        return v


class PotentialHandler(DefaultModel):
    """Base class for storing parametrized force field data"""

    name: str
    expression: str
    independent_variables: Union[str, Set[str]]
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    # Pydantic silently casts some types (int, float, Decimal) to str
    # in models that expect str; this may be updates, see #1098
    @validator("expression", pre=True)
    def is_valid_expr(cls, val):
        if isinstance(val, str):
            return val
        else:
            raise InvalidExpressionError

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "OFFBioTop",
    ) -> None:
        raise NotImplementedError

    def store_potentials(self, parameter_handler: ParameterHandler) -> None:
        raise NotImplementedError

    @requires_package("jax")
    def get_force_field_parameters(self):
        import jax

        params: list = list()
        for potential in self.potentials.values():
            row = [val.magnitude for val in potential.parameters.values()]
            params.append(row)

        return jax.numpy.array(params)

    @requires_package("jax")
    def get_system_parameters(self, p=None):
        import jax

        if p is None:
            p = self.get_force_field_parameters()
        mapping = self.get_mapping()
        q: List = list()

        for key in self.slot_map.keys():
            q.append(p[mapping[self.slot_map[key]]])

        return jax.numpy.array(q)

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
