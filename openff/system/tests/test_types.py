import json

import numpy as np
import pytest
from pydantic import BaseModel

from openff.system import unit
from openff.system.types import ArrayQuantity, DefaultModel, FloatQuantity, UnitArray


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
    def test_float_quantity_model(self):
        class Atom(DefaultModel):
            mass: FloatQuantity["atomic_mass_constant"]
            charge: FloatQuantity["elementary_charge"]
            foo: FloatQuantity
            bar: FloatQuantity["degree"]

        a = Atom(
            mass=4,
            charge=0 * unit.elementary_charge,
            foo=2.0 * unit.nanometer,
            bar="90.0 degree",
        )

        assert a.mass == 4 * unit.atomic_mass_constant
        assert a.charge == 0 * unit.elementary_charge
        assert a.foo == 2.0 * unit.nanometer
        assert a.bar == 90 * unit.degree

        # TODO: Update with custom deserialization to == a.dict()
        assert json.loads(a.json()) == {
            "mass": '{"val": 4, "unit": "atomic_mass_constant"}',
            "charge": '{"val": 0, "unit": "elementary_charge"}',
            "foo": '{"val": 2.0, "unit": "nanometer"}',
            "bar": '{"val": 90.0, "unit": "degree"}',
        }

    def test_array_quantity_model(self):
        class Molecule(DefaultModel):
            masses: ArrayQuantity["atomic_mass_constant"]
            charges: ArrayQuantity["elementary_charge"]
            foo: ArrayQuantity
            bar: ArrayQuantity["degree"]
            baz: ArrayQuantity["second"]

        m = Molecule(
            masses=[16, 1, 1],
            charges=[-1, 0.5, 0.5],
            foo=np.array([2.0, -2.0, 0.0]) * unit.nanometer,
            bar=[0, 90, 180],
            baz=np.array([3, 2, 1]).tobytes(),
        )

        assert json.loads(m.json()) == {
            "masses": '{"val": [16, 1, 1], "unit": "atomic_mass_constant"}',
            "charges": '{"val": [-1.0, 0.5, 0.5], "unit": "elementary_charge"}',
            "foo": '{"val": [2.0, -2.0, 0.0], "unit": "nanometer"}',
            "bar": '{"val": [0, 90, 180], "unit": "degree"}',
            "baz": '{"val": [3, 2, 1], "unit": "second"}',
        }

        parsed = Molecule.parse_raw(m.json())

        # TODO: Better Model __eq__; pydantic just looks at their .dicts, which doesn't
        # play nicely with arrays out of the box
        assert parsed.__fields__ == m.__fields__

        for key in m.dict().keys():
            try:
                assert getattr(m, key) == getattr(parsed, key)
            except ValueError:
                assert all(getattr(m, key) == getattr(parsed, key))
