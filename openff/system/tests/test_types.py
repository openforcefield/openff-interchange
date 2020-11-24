import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from openff.system import unit
from openff.system.types import LengthArray, MassArray, typed_array_encoder


class MolModel(BaseModel):

    length_values: LengthArray
    mass_values: MassArray

    class Config:
        json_encoders = {
            unit.Quantity: typed_array_encoder,
        }


class TestBaseArray:
    def test_pint_model(self):

        model = MolModel(
            length_values=[10, 20] * unit.angstrom,
            mass_values=[1.001, 15.999],
        )

        assert model.length_values.units == unit.nanometer
        assert model.mass_values.units == unit.atomic_mass_constant

        assert np.allclose(model.length_values.m, np.array([1, 2]))
        assert np.allclose(model.mass_values.m, np.array([1.001, 15.999]))

        # TODO: Round-trip
        model.json()

    @pytest.mark.parametrize("default_unit", [unit.second, unit.liter, unit.meter])
    def test_default_units(self, default_unit):
        assert unit.Quantity([1, 1, 1], units=default_unit).units == default_unit

    @pytest.mark.parametrize("input", [0, int, type(None)])
    def test_bad_inputs(self, input):
        with pytest.raises(ValidationError):
            MolModel(
                length_values=input,
                mass_values=[1.001, 15.999],
            )
