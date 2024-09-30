import functools
from collections.abc import Callable
from typing import Annotated, Any

import numpy
from annotated_types import Gt
from openff.toolkit import Quantity
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)

PositiveFloat = Annotated[float, Gt(0)]


def _has_compatible_dimensionality(
    quantity: Quantity,
    unit: str,
    convert: bool,
) -> Quantity:
    """Check if a Quantity has the same dimensionality as a given unit and optionally convert."""
    if quantity.is_compatible_with(unit):
        if convert:
            return quantity.to(unit)
        else:
            return quantity
    else:
        raise ValueError(
            f"Dimensionality of {quantity=} is not compatible with {unit=}",
        )


def _dimensionality_validator_factory(unit: str) -> Callable:
    """Return a function, meant to be passed to a validator, that checks for a specific unit."""
    return functools.partial(_has_compatible_dimensionality, unit=unit, convert=False)


def _unit_validator_factory(unit: str) -> Callable:
    """Return a function, meant to be passed to a validator, that checks for a specific unit."""
    return functools.partial(_has_compatible_dimensionality, unit=unit, convert=True)


(
    _is_distance,
    _is_velocity,
) = (
    _dimensionality_validator_factory(unit=_unit)
    for _unit in [
        "nanometer",
        "nanometer / picosecond",
    ]
)

(
    _is_dimensionless,
    _is_kj_mol,
    _is_nanometer,
    _is_degree,
) = (
    _unit_validator_factory(unit=_unit)
    for _unit in [
        "dimensionless",
        "kilojoule / mole",
        "nanometer",
        "degree",
    ]
)


def quantity_validator(
    value: str | Quantity | dict,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> Quantity:
    """Take Quantity-like objects and convert them to Quantity objects."""
    if info.mode == "json":
        assert isinstance(value, dict), "Quantity must be in dict form here."

        # this is coupled to how a Quantity looks in JSON
        return Quantity(value["val"], value["unit"])

        # some more work may be needed to work with arrays, lists, tuples, etc.

    assert info.mode == "python"

    if isinstance(value, Quantity):
        return value
    elif isinstance(value, str):
        return Quantity(value)
    elif isinstance(value, dict):
        return Quantity(value["val"], value["unit"])
    if "openmm" in str(type(value)):
        from openff.units.openmm import from_openmm

        return from_openmm(value)
    else:
        raise ValueError(f"Invalid type {type(value)} for Quantity")


def quantity_json_serializer(
    quantity: Quantity,
    nxt,
) -> dict:
    """Serialize a Quantity to a JSON-compatible dictionary."""
    magnitude = quantity.m

    if isinstance(magnitude, numpy.ndarray):
        # This could be something fancier, list a bytestring
        magnitude = magnitude.tolist()

    return {
        "val": magnitude,
        "unit": str(quantity.units),
    }


# Pydantic v2 likes to marry validators and serializers to types with Annotated
# https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators
_Quantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    WrapSerializer(quantity_json_serializer),
]

_DimensionlessQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_dimensionless),
    WrapSerializer(quantity_json_serializer),
]

_DistanceQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_distance),
    WrapSerializer(quantity_json_serializer),
]

_LengthQuantity = _DistanceQuantity

_VelocityQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_velocity),
    WrapSerializer(quantity_json_serializer),
]

_DegreeQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_degree),
    WrapSerializer(quantity_json_serializer),
]

_kJMolQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_kj_mol),
    WrapSerializer(quantity_json_serializer),
]


def _is_positions_shape(quantity: Quantity) -> Quantity:
    if quantity.m.shape[1] == 3:
        return quantity
    else:
        raise ValueError(
            f"Quantity {quantity} of wrong shape ({quantity.shape}) to be positions.",
        )


def _duck_to_nanometer(value: Any):
    """Cast list or ndarray without units to Quantity[ndarray] of nanometer."""
    if isinstance(value, (list, numpy.ndarray)):
        return Quantity(value, "nanometer")
    else:
        return value


_PositionsQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_nanometer),
    AfterValidator(_is_positions_shape),
    BeforeValidator(_duck_to_nanometer),
    WrapSerializer(quantity_json_serializer),
]


def _is_box_shape(quantity) -> Quantity:
    if quantity.m.shape == (3, 3):
        return quantity
    elif quantity.m.shape == (3,):
        return numpy.eye(3) * quantity
    else:
        raise ValueError(f"Quantity {quantity} is not a box.")


def _unwrap_list_of_openmm_quantities(value: Any):
    """Unwrap a list of OpenMM quantities to a single Quantity."""
    if isinstance(value, list):
        if any(["openmm" in str(type(element)) for element in value]):
            from openff.units.openmm import from_openmm

            if len({element.unit for element in value}) != 1:
                raise ValueError("All units must be the same.")

            return from_openmm(value)

        else:
            return value

    else:
        return value


_BoxQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_distance),
    AfterValidator(_is_box_shape),
    BeforeValidator(_duck_to_nanometer),
    BeforeValidator(_unwrap_list_of_openmm_quantities),
    WrapSerializer(quantity_json_serializer),
]
