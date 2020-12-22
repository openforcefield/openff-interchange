import json

import pytest
from pydantic import BaseModel

from openff.system import unit
from openff.system.types import DefaultModel, FloatQuantity, UnitArray


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


class TestFloatQuantity:
    def test_unit_model(self):
        class Atom(DefaultModel):
            mass: FloatQuantity["atomic_mass_constant"]
            charge: FloatQuantity["elementary_charge"]
            foo: FloatQuantity

        a = Atom(mass=4, charge=0 * unit.elementary_charge, foo=2.0 * unit.nanometer)

        assert a.mass == 4 * unit.atomic_mass_constant
        assert a.charge == 0 * unit.elementary_charge

        # TODO: Update with custom deserialization to == a.dict()
        assert json.loads(a.json()) == {
            "mass": '{"val": 4, "unit": "atomic_mass_constant"}',
            "charge": '{"val": 0, "unit": "elementary_charge"}',
            "foo": '{"val": 2.0, "unit": "nanometer"}',
        }
