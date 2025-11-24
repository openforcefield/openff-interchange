from openff.toolkit import Topology

from openff.interchange.exceptions import NoPositionsError


class InterchangeTopology(Topology):
    """A subclass of the OpenFF Topology storing no positions."""

    def get_positions(self, *args, **kwargs):
        raise NoPositionsError("Interchange.topology does not store positions. Use Interchange.positions instead.")

    def set_positions(self, *args, **kwargs):
        raise NoPositionsError("Interchange.topology does not store positions. Use Interchange.positions instead.")

    def clear_positions(self, *args, **kwargs):
        raise NoPositionsError("Interchange.topology does not store positions. Use Interchange.positions instead.")
