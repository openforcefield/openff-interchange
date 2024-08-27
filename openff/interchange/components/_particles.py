"""
Temporary module for second-class virtual site objects.
"""

import abc

from openff.toolkit import Quantity

from openff.interchange._annotations import _DistanceQuantity
from openff.interchange.pydantic import _BaseModel


class _VirtualSite(_BaseModel, abc.ABC):
    type: str
    distance: _DistanceQuantity
    orientations: tuple[int, ...]

    @property
    @abc.abstractmethod
    def local_frame_weights(self) -> tuple[list[float], ...]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def local_frame_positions(self) -> Quantity:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def local_frame_coordinates(self) -> Quantity:
        """
        Return the position of this virtual site in its local frame in spherical coordinates.

        The array is of shape (1, 3) and contains `d`, `theta`, and `phi`.

        See Also
        --------
        https://github.com/openforcefield/openff-recharge/blob/0.5.0/openff/recharge/charges/vsite.py#L79-L85

        """
        raise NotImplementedError()
