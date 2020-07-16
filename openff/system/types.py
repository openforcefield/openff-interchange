from . import unit


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
