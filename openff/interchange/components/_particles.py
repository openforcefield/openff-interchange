"""
Temporary module for second-class virtual site objects.
"""

import abc

from openff.models.models import DefaultModel
from openff.models.types.dimension_types import DistanceQuantity
from openff.toolkit import Quantity


class _VirtualSite(DefaultModel, abc.ABC):
    type: str
    distance: DistanceQuantity
    orientations: tuple[int, ...]

    @abc.abstractproperty
    def local_frame_weights(self) -> tuple[list[float], ...]:
        raise NotImplementedError()

    def local_frame_positions(self) -> Quantity:
        raise NotImplementedError()

    @property
    def _local_frame_coordinates(self) -> Quantity:
        """
        Return the position of this virtual site in its local frame in spherical coordinates.

        The array is of shape (1, 3) and contains `d`, `theta`, and `phi`.

        See Also
        --------
        https://github.com/openforcefield/openff-recharge/blob/0.5.0/openff/recharge/charges/vsite.py#L79-L85

        """
        raise NotImplementedError()
