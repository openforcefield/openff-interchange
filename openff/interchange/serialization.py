"""Helpers for serialization/Pydantic things."""

from typing import Annotated

from openff.toolkit import Topology
from pydantic import (
    PlainSerializer,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)


def _topology_custom_before_validator(
    topology: str | Topology,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> Topology:
    if info.mode == "json":
        # Making a new one so no need to deepcopy
        return handler(Topology.from_json(topology))

    assert info.mode == "python"
    if isinstance(topology, Topology):
        return Topology(topology)
    elif isinstance(topology, str):
        return Topology.from_json(topology)
    else:
        raise Exception(f"Failed to convert topology of type {type(topology)}")


def _topology_json_serializer(
    topology: Topology,
    nxt: SerializerFunctionWrapHandler,
) -> str:
    return topology.to_json()


def _topology_dict_serializer(topology: Topology) -> dict:
    return topology.to_dict()


_AnnotatedTopology = Annotated[
    Topology,
    WrapValidator(_topology_custom_before_validator),
    PlainSerializer(_topology_dict_serializer, return_type=dict),
    WrapSerializer(_topology_json_serializer, when_used="json"),
]
