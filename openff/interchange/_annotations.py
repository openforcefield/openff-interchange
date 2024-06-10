import json
from typing import Annotated, Any

import numpy
from openff.toolkit import Quantity
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)


def quantity_validator(
    value: str | Quantity | dict,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> Quantity:
    """Take Quantity-like objects and convert them to Quantity objects."""
    if info.mode == "json":
        if isinstance(value, str):
            value = json.loads(value)

        # this is coupled to how a Quantity looks in JSON
        return Quantity(value["value"], value["unit"])

        # some more work is needed with arrays, lists, tuples, etc.

    assert info.mode == "python"

    if isinstance(value, Quantity):
        return value
    elif isinstance(value, str):
        return Quantity(value)
    elif isinstance(value, dict):
        return Quantity(value["value"], value["unit"])
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
        "value": magnitude,
        "unit": str(quantity.units),
    }


# Pydantic v2 likes to marry validators and serializers to types with Annotated
# https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators
_Quantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    WrapSerializer(quantity_json_serializer),
]


def _is_dimensionless(quantity: Quantity) -> None:
    if quantity.dimensionless:
        return quantity
    else:
        raise ValueError(f"Quantity {quantity} is not dimensionless.")


def _is_distance(quantity: Quantity) -> Quantity:
    if quantity.is_compatible_with("nanometer"):
        return quantity
    else:
        raise ValueError(f"Quantity {quantity} is not a distance.")


def _is_velocity(quantity: Quantity) -> None:
    if quantity.is_compatible_with("nanometer / picosecond"):
        return quantity
    else:
        raise ValueError(f"Quantity {quantity} is not a velocity.")


def _is_degree(quantity: Quantity) -> Quantity:
    try:
        return quantity.to("degree")
    except Exception as error:
        raise ValueError(f"Quantity {quantity} is compatible with degree.") from error


def _is_kj_mol(quantity: Quantity) -> Quantity:
    try:
        return quantity.to("kilojoule / mole")
    except Exception as error:
        raise ValueError("Quantity is not compatible with kJ/mol.") from error


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


def _is_positions(quantity: Quantity) -> Quantity:
    if quantity.m.shape[1] == 3:
        return quantity
    else:
        raise ValueError(
            f"Quantity {quantity} of wrong shape ({quantity.shape}) to be positions.",
        )


def _is_nanometer(quantity: Quantity) -> Quantity:
    try:
        return quantity.to("nanometer")
    except Exception as error:
        raise ValueError(f"Quantity {quantity} is not a distance.") from error


_PositionsQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_nanometer),
    AfterValidator(_is_positions),
    WrapSerializer(quantity_json_serializer),
]


def _is_box(quantity) -> Quantity:
    if quantity.m.shape == (3, 3):
        return quantity
    elif quantity.m.shape == (3,):
        return numpy.eye(3) * quantity
    else:
        raise ValueError(f"Quantity {quantity} is not a box.")


def _duck_to_nanometer(value: Any):
    """Cast list or ndarray without units to Quantity[ndarray] of nanometer."""
    if isinstance(value, (list, numpy.ndarray)):
        return Quantity(value, "nanometer")
    else:
        return value


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
    AfterValidator(_is_box),
    BeforeValidator(_duck_to_nanometer),
    BeforeValidator(_unwrap_list_of_openmm_quantities),
    WrapSerializer(quantity_json_serializer),
]
