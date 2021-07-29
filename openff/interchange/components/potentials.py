import ast
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openff.utilities.utilities import requires_package
from pydantic import Field, PrivateAttr, validator

from openff.interchange.exceptions import MissingParametersError
from openff.interchange.models import DefaultModel, PotentialKey, TopologyKey
from openff.interchange.types import ArrayQuantity, FloatQuantity

if TYPE_CHECKING:
    from openff.interchange.components.mdtraj import OFFBioTop


class Potential(DefaultModel):
    """Base class for storing applied parameters"""

    parameters: Dict[str, FloatQuantity] = dict()
    map_key: Optional[int] = None

    @validator("parameters")
    def validate_parameters(cls, v):
        for key, val in v.items():
            if isinstance(val, list):
                v[key] = ArrayQuantity.validate_type(val)
            else:
                v[key] = FloatQuantity.validate_type(val)
        return v

    def __hash__(self):
        return hash(tuple(self.parameters.values()))


class WrappedPotential(DefaultModel):
    """Model storing other Potential model(s) inside inner data"""

    class InnerData(DefaultModel):
        data: Dict[Potential, float]

    _inner_data: InnerData = PrivateAttr()

    def __init__(self, data):
        if isinstance(data, Potential):
            self._inner_data = self.InnerData(data={data: 1.0})
        elif isinstance(data, dict):
            self._inner_data = self.InnerData(data=data)

    @property
    def parameters(self):
        keys = {
            pot for pot in self._inner_data.data.keys() for pot in pot.parameters.keys()
        }

        params = dict()
        for key in keys:
            sum_ = 0.0
            for pot, coeff in self._inner_data.data.items():
                sum_ += coeff * pot.parameters[key]
            params.update({key: sum_})
        return params

    def __repr__(self):
        return str(self._inner_data.data)


class PotentialHandler(DefaultModel):
    """Base class for storing parametrized force field data"""

    type: str = Field(..., description="The type of potentials this handler stores.")
    expression: str = Field(
        ...,
        description="The analytical expression governing the potentials in this handler.",
    )
    slot_map: Dict[TopologyKey, PotentialKey] = Field(
        dict(),
        description="A mapping between TopologyKey objects and PotentialKey objects.",
    )
    potentials: Dict[PotentialKey, Union[Potential, WrappedPotential]] = Field(
        dict(),
        description="A mapping between PotentialKey objects and Potential objects.",
    )

    @property
    def independent_variables(self) -> Set[str]:
        """Return a set of independent variables, as str, defined as variables in the
        expression that are not found in any potentials."""
        vars_in_potentials = set([*self.potentials.values()][0].parameters.keys())
        vars_in_expression = {
            node.id
            for node in ast.walk(ast.parse(self.expression))
            if isinstance(node, ast.Name)
        }
        return vars_in_expression - vars_in_potentials

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "OFFBioTop",
    ) -> None:
        raise NotImplementedError

    def store_potentials(self, parameter_handler: ParameterHandler) -> None:
        raise NotImplementedError

    def _get_parameters(self, atom_indices: Tuple[int]) -> Dict:
        for topology_key in self.slot_map:
            if topology_key.atom_indices == atom_indices:
                potential_key = self.slot_map[topology_key]
                potential = self.potentials[potential_key]
                parameters = potential.parameters
                return parameters
        raise MissingParametersError(
            f"Could not find parameter in parameter in handler {self.type} "
            f"associated with atoms {atom_indices}"
        )
        pass

    @requires_package("jax")
    def get_force_field_parameters(self):
        """Return a flattened representation of the force field parameters"""
        import jax

        params: list = list()
        for potential in self.potentials.values():
            if isinstance(potential, Potential):
                params.append([val.magnitude for val in potential.parameters.values()])
            elif isinstance(potential, WrappedPotential):
                for inner_pot in potential._inner_data.data.keys():
                    if inner_pot not in params:
                        params.append(
                            [val.magnitude for val in inner_pot.parameters.values()]
                        )

        return jax.numpy.array(params)

    @requires_package("jax")
    def get_system_parameters(self, p=None):
        """Return a flattened representation of system parameters, effectively
        force field parameters as applied to a chemical topology"""
        import jax

        if p is None:
            p = self.get_force_field_parameters()
        mapping = self.get_mapping()
        q: List = list()

        for val in self.slot_map.values():
            if val.bond_order:
                p_ = p[0] * 0.0
                for inner_pot, coeff in self.potentials[val]._inner_data.data.items():
                    p_ += p[mapping[inner_pot]] * coeff
                q.append(p_)
            else:
                q.append(p[mapping[self.potentials[val]]])

        return jax.numpy.array(q)

    def get_mapping(self) -> Dict:
        mapping: Dict = dict()
        idx = 0
        for key, pot in self.potentials.items():
            for p in self.slot_map.values():
                if key == p:
                    if isinstance(pot, Potential):
                        if pot not in mapping:
                            mapping.update({pot: idx})
                            idx += 1
                    elif isinstance(pot, WrappedPotential):
                        for inner_pot in pot._inner_data.data:
                            if inner_pot not in mapping:
                                mapping.update({inner_pot: idx})
                                idx += 1

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
