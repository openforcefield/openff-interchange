import numpy
from openff.toolkit import Quantity

from openff.interchange._annotations import _BoxQuantity
from openff.interchange.models import _BaseModel


class TestBoxQuantity:
    def test_list_cast_to_nanometer_quantity_array(self):
        class M(_BaseModel):
            box: _BoxQuantity

        box = M(box=[2, 3, 4]).box

        assert isinstance(box, Quantity)
        assert str(box.units) == "nanometer"
        assert box.shape == (3, 3)

        numpy.testing.assert_allclose(box, box * numpy.eye(3))
