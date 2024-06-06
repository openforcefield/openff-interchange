import json
from typing import Annotated

from openff.toolkit import Quantity
from pydantic import (
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
    # here is where special cases, like for OpenMM, would go
    else:
        raise ValueError(f"Invalid type {type(value)} for Quantity")


def quantity_json_serializer(
    quantity: Quantity,
    nxt,
) -> dict:
    """Serialize a Quantity to a JSON-compatible dictionary."""
    # Some more work is needed to make arrays play nicely, i.e. not simply doing Quantity.m
    return {
        "value": quantity.m,
        "unit": str(quantity.units),
    }


# Pydantic v2 likes to marry validators and serializers to types with Annotated
# https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators
_Quantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    WrapSerializer(quantity_json_serializer),
]
