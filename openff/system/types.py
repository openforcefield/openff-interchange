import json
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
from pint import Quantity
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.utils import simtk_to_pint


class TypedArrayEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, unit.Quantity):
            if obj.m.shape:
                # Better would be ... "data": np.ascontiguousarray(obj).tobytes().hex()}
                data = {"_nd_": True, "dtype": obj.m.dtype.str, "data": obj.m.tolist()}
                if len(obj.m.shape) > 1:
                    data["shape"] = obj.m.shape
                data["base_unit"] = str(obj.units)
                return data

            else:
                return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def typed_array_encoder(v):
    return json.dumps(v, cls=TypedArrayEncoder)


if TYPE_CHECKING:
    BaseArray = np.ndarray
else:
    ArrayType = TypeVar("ArrayType", bound="BaseArray")

    class _ArrayMeta(type):
        def __getitem__(self, t):
            return type("BaseArray", (BaseArray,), {"__dtype__": t})

    class BaseArray(np.ndarray, metaclass=_ArrayMeta):
        """
        TODO:
          * Can .base_unit be protected?
          * Should this fundamentally be an np.ndarray or unit.Quantity?

        """

        base_unit = "not implemented"

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(
            cls, v: Union[list, int, str, np.ndarray, unit.Quantity]
        ) -> ArrayType:
            if isinstance(v, (int, str)):
                raise TypeError("not implemented")

            # If it's a list, cast into array before asking its __dtype__
            if isinstance(v, list):
                v = np.asarray(v)

            dtype = getattr(cls, "__dtype__", None)
            if isinstance(dtype, tuple):
                dtype, shape = dtype
            else:
                shape = tuple()

            if isinstance(v, omm_unit.Quantity):
                v = simtk_to_pint(v)

            if isinstance(v, Quantity):
                q = v.to(cls.base_unit)
                tmp = q.m
            elif isinstance(v, np.ndarray):
                q = unit.Quantity(v, cls.base_unit)
                tmp = q.m
            else:
                raise ValueError(f"Unexpected type {type(v)} found.")

            try:
                result = np.array(tmp, dtype=dtype, copy=False, ndmin=len(shape))
                if len(shape):
                    result = result.reshape(shape)
                return unit.Quantity(result, cls.base_unit)

            except ValueError:
                raise ValueError("Could not cast {} to NumPy Array!".format(v))

        def __repr__(self):
            return str(self) + " " + self.base_unit


class LengthArray(BaseArray):
    base_unit = "nanometer"


class MassArray(BaseArray):
    base_unit = "atomic_mass_constant"


class Meta(type):
    def __repr__(cls):
        return "ok" + cls.base_unit


class BaseQuantity(float, metaclass=Meta):
    base_unit = "not implemented"

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Union[str, int, float]):
        if isinstance(v, str):
            q = unit.Quantity(v)
            return cls(q.to(cls.base_unit).m)
        else:
            return cls(float(v))

    def __repr__(self):
        return str(self) + " " + self.base_unit


class SigmaQuantity(BaseQuantity):
    base_unit = "nanometer"


class EpsilonQuantity(BaseQuantity):
    base_unit = "kcal/mol"
