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

from openff.interchange.common._topology import InterchangeTopology


def _topology_custom_before_validator(
    topology: str | Topology | InterchangeTopology,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> InterchangeTopology:
    if info.mode == "json":
        # Making a new one so no need to deepcopy
        return InterchangeTopology.from_json(topology)

    assert info.mode == "python"
    if isinstance(topology, Topology | InterchangeTopology):
        return InterchangeTopology(topology)
    elif isinstance(topology, str):
        return InterchangeTopology.from_json(topology)
    else:
        raise Exception(f"Failed to convert topology of type {type(topology)}")


def _topology_json_serializer(
    topology: InterchangeTopology,
    nxt: SerializerFunctionWrapHandler,
) -> str:
    return topology.to_json()


def _topology_dict_serializer(topology: InterchangeTopology) -> dict:
    return topology.to_dict()


_AnnotatedTopology = Annotated[
    InterchangeTopology,
    WrapValidator(_topology_custom_before_validator),
    PlainSerializer(_topology_dict_serializer, return_type=dict),
    WrapSerializer(_topology_json_serializer, when_used="json"),
]
