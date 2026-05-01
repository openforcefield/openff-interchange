"""Pydantic base model with custom settings."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class _BaseModel(BaseModel):
    """A custom Pydantic model used by other components."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        # serialize_as_any=True breaks pint.Quantity serialization in pydantic >=2.12
        # (pydantic/pydantic#12348); the Annotated WrapSerializer on _Quantity handles
        # JSON serialization correctly without it
        return super().model_dump_json(**kwargs)
