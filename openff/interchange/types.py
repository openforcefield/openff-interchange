"""Custom models for dealing with unit-bearing quantities in a Pydantic-compatible manner."""
import json
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from openff.units import unit
from openff.utilities.utilities import has_package, requires_package
from openmm import unit as openmm_unit

from openff.interchange.exceptions import (
    MissingUnitError,
    UnitValidationError,
    UnsupportedExportError,
)

if TYPE_CHECKING or has_package("unyt"):
    import unyt


class _FloatQuantityMeta(type):
    def __getitem__(self, t):
        return type("FloatQuantity", (FloatQuantity,), {"__unit__": t})


class FloatQuantity(float, metaclass=_FloatQuantityMeta):
    """A model for unit-bearing floats."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        """Process a value tagged with units into one tagged with "OpenFF" style units."""
        unit_ = getattr(cls, "__unit__", Any)
        if unit_ is Any:
            if isinstance(val, (float, int)):
                # TODO: Can this exception be raised with knowledge of the field it's in?
                raise MissingUnitError(f"Value {val} needs to be tagged with a unit")
            elif isinstance(val, unit.Quantity):
                return unit.Quantity(val)
            elif isinstance(val, openmm_unit.Quantity):
                return _from_omm_quantity(val)
            else:
                raise UnitValidationError(
                    f"Could not validate data of type {type(val)}"
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
            if isinstance(val, openmm_unit.Quantity):
                return _from_omm_quantity(val).to(unit_)
            if has_package("unyt"):
                if isinstance(val, unyt.unyt_quantity):
                    return _from_unyt_quantity(val).to(unit_)
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                return val * unit_
            if isinstance(val, str):
                # could do custom deserialization here?
                return unit.Quantity(val).to(unit_)
            raise UnitValidationError(f"Could not validate data of type {type(val)}")


def _from_omm_quantity(val: openmm_unit.Quantity):
    """
    Convert float or array quantities tagged with SimTK/OpenMM units to a Pint-compatible quantity.
    """
    unit_ = val.unit
    val_ = val.value_in_unit(unit_)
    if type(val_) in {float, int}:
        unit_ = val.unit
        return val_ * unit.Unit(str(unit_))
    elif type(val_) in {tuple, list, np.ndarray}:
        array = np.asarray(val_)
        return array * unit.Unit(str(unit_))
    elif isinstance(val_, (float, int)) and type(val_).__module__ == "numpy":
        return val_ * unit.Unit(str(unit_))
    else:
        raise UnitValidationError(
            "Found a openmm.unit.Unit wrapped around something other than a float-like "
            f"or np.ndarray-like. Found a unit wrapped around type {type(val_)}."
        )


@requires_package("unyt")
def _from_unyt_quantity(val: "unyt.unyt_array"):
    """Convert unyt arrays to Pint quantities."""
    quantity = val.to_pint()
    # Ensure a float-like quantity is a float, not a scalar array
    if isinstance(val, unyt.unyt_quantity):
        quantity = float(quantity.magnitude) * quantity.units
    return quantity


class QuantityEncoder(json.JSONEncoder):
    """
    JSON encoder for unit-wrapped floats and NumPy arrays.

    This is intended to operate on FloatQuantity and ArrayQuantity objects.
    """

    def default(self, obj):  # noqa
        if isinstance(obj, unit.Quantity):
            if isinstance(obj.magnitude, (float, int)):
                data = obj.magnitude
            elif isinstance(obj.magnitude, np.ndarray):
                data = obj.magnitude.tolist()
            else:
                # This shouldn't ever be hit if our object models
                # behave in ways we expect?
                raise UnsupportedExportError(
                    f"trying to serialize unsupported type {type(obj.magnitude)}"
                )
            return {
                "val": data,
                "unit": str(obj.units),
            }


def custom_quantity_encoder(v):
    """Wrap json.dump to use QuantityEncoder."""
    return json.dumps(v, cls=QuantityEncoder)


def json_loader(data: str) -> dict:
    """Load JSON containing custom unit-tagged quantities."""
    # TODO: recursively call this function for nested models
    out: Dict = json.loads(data)
    for key, val in out.items():
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
        out[key] = unit_ * val
    return out


class _ArrayQuantityMeta(type):
    def __getitem__(self, t):
        return type("ArrayQuantity", (ArrayQuantity,), {"__unit__": t})


if TYPE_CHECKING:
    ArrayQuantity = np.ndarray
else:

    class ArrayQuantity(float, metaclass=_ArrayQuantityMeta):
        """A model for unit-bearing arrays."""

        @classmethod
        def __get_validators__(cls):
            yield cls.validate_type

        @classmethod
        def validate_type(cls, val):
            """Process an array tagged with units into one tagged with "OpenFF" style units."""
            unit_ = getattr(cls, "__unit__", Any)
            if unit_ is Any:
                if isinstance(val, (list, np.ndarray)):
                    # TODO: Can this exception be raised with knowledge of the field it's in?
                    raise MissingUnitError(
                        f"Value {val} needs to be tagged with a unit"
                    )
                elif isinstance(val, unit.Quantity):
                    # Redundant cast? Maybe this handles pint vs openff.interchange.unit?
                    return unit.Quantity(val)
                elif isinstance(val, openmm_unit.Quantity):
                    return _from_omm_quantity(val)
                else:
                    raise UnitValidationError(
                        f"Could not validate data of type {type(val)}"
                    )
            else:
                unit_ = unit(unit_)
                if isinstance(val, unit.Quantity):
                    assert unit_.dimensionality == val.dimensionality
                    return val.to(unit_)
                if isinstance(val, openmm_unit.Quantity):
                    return _from_omm_quantity(val).to(unit_)
                if isinstance(val, (np.ndarray, list)):
                    if has_package("unyt"):
                        # Must check for unyt_array, not unyt_quantity, which is a subclass
                        if isinstance(val, unyt.unyt_array):
                            return _from_unyt_quantity(val).to(unit_)
                        else:
                            return val * unit_
                    else:
                        return val * unit_
                if isinstance(val, bytes):
                    # Define outside loop
                    dt = np.dtype(int)
                    dt.newbyteorder("<")
                    return np.frombuffer(val, dtype=dt) * unit_
                if isinstance(val, str):
                    # could do custom deserialization here?
                    raise NotImplementedError
                    #  return unit.Quantity(val).to(unit_)
                raise UnitValidationError(
                    f"Could not validate data of type {type(val)}"
                )
