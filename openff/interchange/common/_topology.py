from openff.toolkit import Topology


def validate_topology(value) -> Topology | None:
    """Validate a topology-like argument, spliced from a previous validator."""
    from openff.interchange.exceptions import InvalidTopologyError

    if value is None:
        return None
    if isinstance(value, Topology):
        return Topology(other=value)
    elif isinstance(value, list):
        return Topology.from_molecules(value)
    elif type(value) is dict:
        return Topology.from_dict(value)
    elif type(value) is str:
        return Topology.from_json(value)
    else:
        raise InvalidTopologyError(
            "Could not process topology argument, expected openff.toolkit.Topology. "
            f"Found object of type {type(value)}.",
        )
