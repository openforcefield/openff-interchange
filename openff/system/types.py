import json
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from openff.system import unit


class _FloatQuantityMeta(type):
    def __getitem__(self, t):
        return type("FloatQuantity", (FloatQuantity,), {"__unit__": t})


class FloatQuantity(float, metaclass=_FloatQuantityMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        unit_ = getattr(cls, "__unit__", Any)
        if unit_ is Any:
            if isinstance(val, (float, int)):
                # TODO: Can this exception be raised with knowledge of the field it's in?
                raise ValueError(f"Value {val} needs to be tagged with a unit")
            elif isinstance(val, unit.Quantity):
                return unit.Quantity(val)
            else:
                raise ValueError(f"Could not validate data of type {type(val)}")
        else:
            unit_ = unit(unit_)
            if isinstance(val, unit.Quantity):
                # some custom behavior could go here
                assert unit_.dimensionality == val.dimensionality
                # return through converting to some intended default units (taken from the class)
                return val.to(unit_)
                # could return here, without converting
                # (could be inconsistent with data model - heteregenous but compatible units)
                # return val
            elif isinstance(val, (float, int)) and not isinstance(val, bool):
                return val * unit_
            elif isinstance(val, str):
                # could do custom deserialization here?
                return unit.Quantity(val).to(unit_)
            else:
                raise ValueError(f"Could not validate data of type {type(val)}")


class QuantityEncoder(json.JSONEncoder):
    """JSON encoder for unit-wrapped floats and np arrays. Should work
    for both FloatQuantity and ArrayQuantity"""

    def default(self, obj):
        if isinstance(obj, unit.Quantity):
            if isinstance(obj.magnitude, (float, int)):
                data = obj.magnitude
            elif isinstance(obj.magnitude, np.ndarray):
                data = obj.magnitude.tolist()
            else:
                # This shouldn't ever be hit if our object models
                # behave in ways we expect?
                raise Exception(
                    f"trying to serialize unsupported type {type(obj.magnitude)}"
                )
            return {
                "val": data,
                "unit": str(obj.units),
            }


def custom_quantity_encoder(v):
    return json.dumps(v, cls=QuantityEncoder)


def json_loader(data: str) -> dict:
    # TODO: recursively call this function for nested models
    data = json.loads(data)
    for key, val in data.items():
        try:
            # Directly look for an encoded FloatQuantity/ArrayQuantity,
            # which is itself a dict
            v = json.loads(val)
        except json.JSONDecodeError:
            # Handles some cases of the val being a primitive type
            continue
        # TODO: More gracefully parse non-FloatQuantity/ArrayQuantity dicts
        unit_ = unit(v["unit"])
        val = v["val"]
        data[key] = unit_ * val
    return data


class _ArrayQuantityMeta(type):
    def __getitem__(self, t):
        return type("ArrayQuantity", (ArrayQuantity,), {"__unit__": t})


if TYPE_CHECKING:
    Array = np.array
else:

    class ArrayQuantity(float, metaclass=_ArrayQuantityMeta):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate_type

        @classmethod
        def validate_type(cls, val):
            unit_ = getattr(cls, "__unit__", Any)
            if unit_ is Any:
                if isinstance(val, (list, np.ndarray)):
                    # TODO: Can this exception be raised with knowledge of the field it's in?
                    raise ValueError(f"Value {val} needs to be tagged with a unit")
                elif isinstance(val, unit.Quantity):
                    # Redundant cast? Maybe this handles pint vs openff.system.unit?
                    return unit.Quantity(val)
                else:
                    raise ValueError(f"Could not validate data of type {type(val)}")
            else:
                unit_ = unit(unit_)
                if isinstance(val, unit.Quantity):
                    assert unit_.dimensionality == val.dimensionality
                    return val.to(unit_)
                elif isinstance(val, (np.ndarray, list)):
                    return val * unit_
                elif isinstance(val, bytes):
                    # Define outside loop
                    dt = np.dtype(int)
                    dt.newbyteorder("<")
                    return np.frombuffer(val, dtype=dt) * unit_
                elif isinstance(val, str):
                    # could do custom deserialization here?
                    raise NotImplementedError
                    #  return unit.Quantity(val).to(unit_)
                else:
                    raise ValueError(f"Could not validate data of type {type(val)}")


class DefaultModel(BaseModel):
    class Config:
        json_encoders = {
            unit.Quantity: custom_quantity_encoder,
        }
        json_loads = json_loader
        validate_assignment = True
        arbitrary_types_allowed = True
