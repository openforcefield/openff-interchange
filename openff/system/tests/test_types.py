import numpy as np
import pytest
from pydantic import BaseModel

from openff.system import unit
from openff.system.types import BaseArray


class TimeArray(BaseArray):
    base_unit = "year"


class DistanceArray(BaseArray):
    base_unit = "meter"


class UnitModel(BaseModel):

    distance_values: DistanceArray
    time_values: TimeArray


class TestBaseArray:
    def test_pint_model(self):

        model = UnitModel(
            distance_values=[100, 200] * unit.cm,
            time_values=[12.0, 3.0] * unit.month,
        )

        assert model.distance_values.units == unit.meter
        assert model.time_values.units == unit.year

        assert np.allclose(model.distance_values.m, np.array([1, 2]))
        assert np.allclose(model.time_values.m, np.array([1, 0.25]))

    @pytest.mark.parametrize("default_unit", [unit.second, unit.liter, unit.meter])
    def test_default_units(self, default_unit):
        assert unit.Quantity([1, 1, 1], units=default_unit).units == default_unit

    @pytest.mark.parametrize("input", [0, int, type(None)])
    def test_bad_inputs(self, input):
        with pytest.raises(TypeError):
            unit.Quantity([4, 4], units=input)
