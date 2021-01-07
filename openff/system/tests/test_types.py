import json

import numpy as np
import pytest
from pydantic import ValidationError

from openff.system import unit
from openff.system.types import ArrayQuantity, DefaultModel, FloatQuantity


class TestQuantityTypes:
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

        parsed = Atom.parse_raw(a.json())
        assert a == parsed

        assert Atom(**a.dict()) == a

    @pytest.mark.parametrize("val", [True, [1]])
    def test_bad_float_quantity_type(self, val):
        class Atom(DefaultModel):
            mass: FloatQuantity["atomic_mass_constant"]

        with pytest.raises(
            ValueError, match=r"Could not validate data of type .*[bool|list].*"
        ):
            Atom(mass=True)

    def test_array_quantity_model(self):
        class Molecule(DefaultModel):
            masses: ArrayQuantity["atomic_mass_constant"]
            charges: ArrayQuantity["elementary_charge"]
            foo: ArrayQuantity
            bar: ArrayQuantity["degree"]
            baz: ArrayQuantity["second"]

        m = Molecule(
            masses=[16, 1, 1],
            charges=np.asarray([-1, 0.5, 0.5]),
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

    @pytest.mark.parametrize("val", [True, [1]])
    def test_bad_array_quantity_type(self, val):
        class Atom(DefaultModel):
            mass: ArrayQuantity["atomic_mass_constant"]

        with pytest.raises(
            ValueError, match=r"Could not validate data of type .*[bool|list].*"
        ):
            Atom(mass=True)

    def test_float_and_quantity_type(self):
        class MixedModel(DefaultModel):
            scalar_data: FloatQuantity
            array_data: ArrayQuantity
            name: str

        m = MixedModel(
            scalar_data=1.0 * unit.meter, array_data=[-1, 0] * unit.second, name="foo"
        )

        assert json.loads(m.json()) == {
            "scalar_data": '{"val": 1.0, "unit": "meter"}',
            "array_data": '{"val": [-1, 0], "unit": "second"}',
            "name": "foo",
        }

        parsed = MixedModel.parse_raw(m.json())

        for key in m.dict().keys():
            try:
                assert getattr(m, key) == getattr(parsed, key)
            except ValueError:
                assert all(getattr(m, key) == getattr(parsed, key))

    def test_model_mutability(self):
        class Model(DefaultModel):
            time: FloatQuantity["second"]
            lengths: ArrayQuantity["foot"]

        m = Model(time=10 * unit.second, lengths=[3, 4] * unit.foot)

        m.time = 0.5 * unit.minute
        m.lengths = [2, 1] * unit.yard

        assert m.time == 30 * unit.second
        assert (m.lengths == [6, 3] * unit.foot).all()

        with pytest.raises(ValidationError, match="1 validation error for Model"):
            m.time = 1 * unit.gram

        with pytest.raises(ValidationError, match="1 validation error for Model"):
            m.lengths = 1 * unit.watt
