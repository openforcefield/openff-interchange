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
        return super().model_dump(
            include=self.__class__.model_fields.keys(),
            serialize_as_any=True,
            **kwargs,
        )

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(
            include=self.__class__.model_fields.keys(),
            serialize_as_any=True,
            **kwargs,
        )
