import pytest
from pydantic import BaseModel

from openff.system import unit
from openff.system.types import UnitArray


class UnitModel(BaseModel):

    any_values: UnitArray
    distance_values: UnitArray[unit.nm]
    time_values: UnitArray[unit.ns]


class TestUnitArray:
    def test_pint_model(self):

        model = UnitModel(
            any_values=[4, 2, 1] * unit.year,
            distance_values=[1, 2] * unit.nm,
            time_values=[2.0, 4.0] * unit.ns,
        )

        assert model.any_values.units == unit.year
        assert model.distance_values.units == unit.nm
        assert model.time_values.units == unit.ns

    @pytest.mark.parametrize("default_unit", [unit.second, unit.liter, unit.meter])
    def test_default_units(self, default_unit):
        assert UnitArray([1, 1, 1], units=default_unit).units == default_unit

    @pytest.mark.parametrize("input", [0, int, type(None)])
    def test_bad_inputs(self, input):
        with pytest.raises(TypeError):
            UnitArray([4, 4], units=input)
