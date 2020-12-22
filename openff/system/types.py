import json
from typing import Any

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
            if isinstance(val, float):
                # input doesn't have a unit, it's just a float-ish type
                raise ValueError("This needs unit")
            elif isinstance(val, unit.Quantity):
                return unit.Quantity(val)
            else:
                raise ValueError(
                    f"Bad input, expected float or pint.Quantity-like, got {type(val)})"
                )
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
            elif isinstance(val, (float, int)):
                return val * unit_
            elif isinstance(val, str):
                # could do custom deserialization here?
                return unit.Quantity(val).to(unit_)


class FloatQuantityEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, unit.Quantity):
            return {
                "val": obj.magnitude,
                "unit": str(obj.units),
            }
        return json.JSONEncoder.default(self, obj)


def custom_quantity_encoder(v):
    return json.dumps(v, cls=FloatQuantityEncoder)


class DefaultModel(BaseModel):
    class Config:
        json_encoders = {unit.Quantity: custom_quantity_encoder}


class UnitArrayMeta(type):
    # TODO: would be nice to be able to sneak dtype and/or units in here
    def __getitem__(self, units, dtype=None):
        return type("UnitArray", (UnitArray,), {"_dtype": dtype, "_units": units})


class UnitArray(unit.Quantity, metaclass=UnitArrayMeta):
    """
    Thin wrapper around pint.Quantity for compliance with Pydantic classes

    See https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-594639970
    """

    # TODO: Handle various cases of implicit units, i.e. NumPy arrays that intend
    # TODO: Use dtype

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        try:
            val = unit.Quantity(val)
            return val
        # TODO: Handle other exceptions, like pint.UndefinedUnitError
        except TypeError as e:
            raise TypeError from e
