from openff.toolkit import Topology


class InterchangeTopology(Topology):
    """A subclass of the OpenFF Topology storing no positions."""

    def get_positions(self, *args, **kwargs):
        raise NotImplementedError("Interchange.topology does not store positions. Use Interchange.positions instead.")

    def set_positions(self, *args, **kwargs):
        raise NotImplementedError("Interchange.topology does not store positions. Use Interchange.positions instead.")

    def clear_positions(self, *args, **kwargs):
        raise NotImplementedError("Interchange.topology does not store positions. Use Interchange.positions instead.")
