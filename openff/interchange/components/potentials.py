"""Models for storing applied force field parameters."""

from __future__ import annotations

import ast
import functools
import gc
import json
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

import numpy
from numpy.typing import ArrayLike
from openff.toolkit import Quantity
from openff.utilities.utilities import has_package, requires_package
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
)
from pydantic.functional_validators import WrapValidator

from openff.interchange._annotations import _Quantity
from openff.interchange.exceptions import MissingParametersError
from openff.interchange.models import (
    LibraryChargeTopologyKey,
    PotentialKey,
    TopologyKey,
)
from openff.interchange.pydantic import _BaseModel

if TYPE_CHECKING:
    if has_package("jax"):
        from jax import Array
    else:
        Array: TypeAlias = Any  # type: ignore[no-redef]
else:
    Array: TypeAlias = ArrayLike


class Potential(_BaseModel):
    """Base class for storing applied parameters."""

    parameters: dict[str, _Quantity] = Field(dict())
    map_key: int | None = None

    def __hash__(self) -> int:
        return hash(tuple(self.parameters.values()))


class WrappedPotential(_BaseModel):
    """Model storing other Potential model(s) inside inner data."""

    _inner_data: dict[Potential, float] = PrivateAttr()

    def __init__(self, data: Potential | dict) -> None:
        # Needed to set some Pydantic magic, at least __pydantic_private__;
        # won't actually process the input here
        super().__init__()

        if isinstance(data, Potential):
            data = {data: 1.0}

        self._inner_data = data

    @property
    def parameters(self) -> dict[str, Quantity]:
        """Get the parameters as represented by the stored potentials and coefficients."""
        keys: set[str] = {param_key for pot in self._inner_data.keys() for param_key in pot.parameters.keys()}
        params = dict()
        for key in keys:
            params.update(
                {
                    key: sum(coeff * pot.parameters[key] for pot, coeff in self._inner_data.items()),
                },
            )
        return params

    def __repr__(self) -> str:
        return str(self._inner_data)


def validate_potential_or_wrapped_potential(
    v: Any,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> Potential | WrappedPotential:
    """Validate the parameters field of a Potential object."""
    if info.mode == "json":
        if "parameters" in v:
            return Potential.model_validate(v)
        else:
            return WrappedPotential.model_validate(v)
    else:
        raise NotImplementedError(f"Validation mode {info.mode} not implemented.")


PotentialOrWrappedPotential = Annotated[
    Potential | WrappedPotential,
    WrapValidator(validate_potential_or_wrapped_potential),
]


def validate_key_map(v: Any, handler, info) -> dict:
    """Validate the key_map field of a Collection object."""
    from openff.interchange.models import (
        AngleKey,
        BondKey,
        ImproperTorsionKey,
        LibraryChargeTopologyKey,
        ProperTorsionKey,
        SingleAtomChargeTopologyKey,
    )

    tmp = dict()
    if info.mode in ("json", "python"):
        for key, val in v.items():
            val_dict = json.loads(val)

            key_class: type[TopologyKey]

            match val_dict["associated_handler"]:
                case "Bonds":
                    key_class = BondKey
                case "Angles":
                    key_class = AngleKey
                case "ProperTorsions":
                    key_class = ProperTorsionKey
                case "ImproperTorsions":
                    key_class = ImproperTorsionKey
                case "LibraryCharges":
                    key_class = LibraryChargeTopologyKey  # type: ignore[assignment]
                case (
                    "ToolkitAM1BCCHandler"
                    | "molecules_with_preset_charges"
                    | "NAGLChargesHandler"
                    | "ChargeIncrementModelHandler"
                ):
                    key_class = SingleAtomChargeTopologyKey  # type: ignore[assignment]
                case _:
                    key_class = TopologyKey

            try:
                tmp.update(
                    {
                        key_class.model_validate_json(
                            key,
                        ): PotentialKey.model_validate_json(val),
                    },
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to deserialize a `PotentialKey` with {val_dict['associated_handler']=}",
                ) from error

            del key_class

        v = tmp

    else:
        raise ValueError(f"Validation mode {info.mode} not implemented.")

    return v


def serialize_key_map(
    value: dict[TopologyKey, PotentialKey],
    handler,
    info,
) -> dict[str, str]:
    """Serialize the parameters field of a Potential object."""
    if info.mode == "json":
        return {key.model_dump_json(): value.model_dump_json() for key, value in value.items()}

    else:
        raise NotImplementedError(f"Serialization mode {info.mode} not implemented.")


KeyMap = Annotated[
    dict[TopologyKey | LibraryChargeTopologyKey, PotentialKey],
    WrapValidator(validate_key_map),
    WrapSerializer(serialize_key_map),
]


def validate_potential_dict(
    v: Any,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
):
    """Validate the parameters field of a Potential object."""
    if info.mode == "json":
        return {PotentialKey.model_validate_json(key): Potential.model_validate_json(val) for key, val in v.items()}

    elif info.mode == "python":
        # Unclear why str sometimes sneak into here in Python mode; everything
        # should be object (PotentialKey/Potential) or dict at this point ...
        return {
            PotentialKey.model_validate_json(key) if isinstance(key, str) else key: (
                Potential.model_validate_json(val) if isinstance(val, str) else val
            )
            for key, val in v.items()
        }

    else:
        raise NotImplementedError(f"Validation mode {info.mode} not implemented.")


def serialize_potentials(
    value: dict[PotentialKey, Potential],
    handler,
    info,
) -> dict[str, str]:
    """Serialize the potentials field."""
    if info.mode == "json":
        return {key.model_dump_json(): _value.model_dump_json() for key, _value in value.items()}

    else:
        raise NotImplementedError(f"Serialization mode {info.mode} not implemented.")


Potentials = Annotated[
    dict[PotentialKey, PotentialOrWrappedPotential],
    WrapValidator(validate_potential_dict),
    WrapSerializer(serialize_potentials),
]


class Collection(_BaseModel):
    """Base class for storing parametrized force field data."""

    type: str = Field(..., description="The type of potentials this handler stores.")
    is_plugin: bool = Field(
        False,
        description="Whether this collection is defined as a plugin.",
    )
    expression: str = Field(
        ...,
        description="The analytical expression governing the potentials in this handler.",
    )
    key_map: KeyMap = Field(
        dict(),
        description="A mapping between TopologyKey objects and PotentialKey objects.",
    )
    potentials: Potentials = Field(
        dict(),
        description="A mapping between PotentialKey objects and Potential objects.",
    )

    @property
    def independent_variables(self) -> set[str]:
        """
        Return a set of variables found in the expression but not in any potentials.
        """
        vars_in_potentials = next(iter(self.potentials.values())).parameters.keys()
        vars_in_expression = {node.id for node in ast.walk(ast.parse(self.expression)) if isinstance(node, ast.Name)}
        return vars_in_expression - vars_in_potentials

    def _get_parameters(self, atom_indices: tuple[int]) -> dict:
        for topology_key in self.key_map:
            if topology_key.atom_indices == atom_indices:
                potential_key = self.key_map[topology_key]
                potential = self.potentials[potential_key]
                parameters = potential.parameters
                return parameters
        raise MissingParametersError(
            f"Could not find parameter in parameter in handler {self.type} associated with atoms {atom_indices}",
        )

    def get_force_field_parameters(
        self,
        use_jax: bool = False,
    ) -> ArrayLike | Array:
        """Return a flattened representation of the force field parameters."""
        # TODO: Handle WrappedPotential
        if any(isinstance(potential, WrappedPotential) for potential in self.potentials.values()):
            raise NotImplementedError

        if use_jax:
            from jax import numpy as jax_numpy

            return jax_numpy.array(
                [[v.m for v in p.parameters.values()] for p in self.potentials.values()],
            )
        else:
            return numpy.array(
                [[v.m for v in p.parameters.values()] for p in self.potentials.values()],
            )

    def set_force_field_parameters(self, new_p: ArrayLike) -> None:
        """Set the force field parameters from a flattened representation."""

        # Clear cache of all methods that are wrapped by functools.lru_cache
        # A better solution might be at the level of parameter re-assignment
        # See issue #1234
        for obj in gc.get_objects():
            if isinstance(obj, functools._lru_cache_wrapper) and obj.__module__.startswith("openff.interchange"):
                obj.cache_clear()

        mapping = self.get_mapping()
        if new_p.shape[0] != len(mapping):  # type: ignore
            raise RuntimeError

        for potential_key, potential_index in self.get_mapping().items():
            potential = self.potentials[potential_key]
            if len(new_p[potential_index, :]) != len(potential.parameters):  # type: ignore
                raise RuntimeError

            for parameter_index, parameter_key in enumerate(potential.parameters):
                parameter_units = potential.parameters[parameter_key].units
                modified_parameter = new_p[potential_index, parameter_index]  # type: ignore

                self.potentials[potential_key].parameters[parameter_key] = modified_parameter * parameter_units

    def get_system_parameters(
        self,
        p=None,
        use_jax: bool = False,
    ) -> ArrayLike | Array:
        """
        Return a flattened representation of system parameters.

        These values are effectively force field parameters as applied to a chemical topology.
        """
        # TODO: Handle WrappedPotential
        if any(isinstance(potential, WrappedPotential) for potential in self.potentials.values()):
            raise NotImplementedError

        if p is None:
            p = self.get_force_field_parameters(use_jax=use_jax)
        mapping = self.get_mapping()

        q: list = list()
        for potential_key in self.key_map.values():
            index = mapping[potential_key]
            q.append(p[index])

        if use_jax:
            from jax import numpy as jax_numpy

            return jax_numpy.array(q)
        else:
            return numpy.array(q)

    def get_mapping(self) -> dict[PotentialKey, int]:
        """Get a mapping between potentials and array indices."""
        mapping: dict = dict()
        index = 0
        for potential_key in self.key_map.values():
            if potential_key not in mapping:
                mapping[potential_key] = index
                index += 1

        return mapping

    def parametrize(
        self,
        p=None,
        use_jax: bool = True,
    ) -> ArrayLike | Array:
        """Return an array of system parameters, given an array of force field parameters."""
        if p is None:
            p = self.get_force_field_parameters(use_jax=use_jax)

        return self.get_system_parameters(p=p, use_jax=use_jax)

    def parametrize_partial(self):
        """Return a function that will call `self.parametrize()` with arguments specified by `self.mapping`."""
        from functools import partial

        return partial(
            self.parametrize,
            mapping=self.get_mapping(),
        )

    @requires_package("jax")
    def get_param_matrix(self) -> ArrayLike | Array:
        """Get a matrix representing the mapping between force field and system parameters."""
        from functools import partial

        import jax

        p = self.get_force_field_parameters(use_jax=True)

        parametrize_partial = partial(
            self.parametrize,
        )

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])  # type: ignore[union-attr]

    def __getitem__(self, key) -> Potential | WrappedPotential:
        if isinstance(key, tuple) and key not in self.key_map and tuple(reversed(key)) in self.key_map:
            return self.potentials[self.key_map[tuple(reversed(key))]]  # type: ignore[index]

        return self.potentials[self.key_map[key]]

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return self is other


def validate_collections(
    v: Any,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> dict:
    """Validate the collections dict from a JSON blob."""
    from openff.interchange.smirnoff import (
        SMIRNOFFAngleCollection,
        SMIRNOFFBondCollection,
        SMIRNOFFCollection,
        SMIRNOFFConstraintCollection,
        SMIRNOFFElectrostaticsCollection,
        SMIRNOFFImproperTorsionCollection,
        SMIRNOFFProperTorsionCollection,
        SMIRNOFFvdWCollection,
        SMIRNOFFVirtualSiteCollection,
    )

    _class_mapping: dict[str, type[SMIRNOFFCollection]] = {
        "Bonds": SMIRNOFFBondCollection,
        "Angles": SMIRNOFFAngleCollection,
        "Constraints": SMIRNOFFConstraintCollection,
        "ProperTorsions": SMIRNOFFProperTorsionCollection,
        "ImproperTorsions": SMIRNOFFImproperTorsionCollection,
        "vdW": SMIRNOFFvdWCollection,
        "Electrostatics": SMIRNOFFElectrostaticsCollection,
        "VirtualSites": SMIRNOFFVirtualSiteCollection,
    }

    if info.mode in ("json", "python"):
        return {
            collection_name: _class_mapping[collection_name].model_validate(
                collection_data,
            )
            for collection_name, collection_data in v.items()
        }
    else:
        raise ValueError(f"Validation mode {info.mode} not implemented.")


_AnnotatedCollections = Annotated[
    dict[str, Collection],
    WrapValidator(validate_collections),
]
