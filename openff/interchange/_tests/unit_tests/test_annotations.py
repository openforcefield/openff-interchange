import numpy
import pytest
from openff.toolkit import Quantity
from pydantic import ValidationError

from openff.interchange._annotations import _BoxQuantity, _kJMolQuantity, _MassQuantity
from openff.interchange.models import _BaseModel


class TestAnnotations:
    def test_dimensionality_check_fails(self):
        class Model(_BaseModel):
            mass: _MassQuantity

        with pytest.raises(
            ValidationError,
            match=r"Dimensionality.*is not compatible with",
        ):
            Model(mass=Quantity(400.0, "kelvin"))

    def test_dimensionality_check_soft_passes(self):
        class Model(_BaseModel):
            mass: _MassQuantity

        assert Model(mass=Quantity(1e-3 / 6.02214076e23, "kg")).mass.m_as("amu") == pytest.approx(1.0)

    def test_unit_check_coerced(self):
        class Model(_BaseModel):
            dde: _kJMolQuantity

        assert Model(dde=Quantity(1.5, "kcal/mol")).dde.units == "kilojoule / mole"

    def test_dimensionality_check_passes(self):
        class Model(_BaseModel):
            dde: _kJMolQuantity

        assert Model(dde=Quantity(1.5, "kJ/mol")).dde.m_as("kJ/mol") == pytest.approx(1.5)


class TestBoxQuantity:
    def test_list_cast_to_nanometer_quantity_array(self):
        class M(_BaseModel):
            box: _BoxQuantity

        box = M(box=[2, 3, 4]).box

        assert isinstance(box, Quantity)
        assert str(box.units) == "nanometer"
        assert box.shape == (3, 3)

        numpy.testing.assert_allclose(box, box * numpy.eye(3))
