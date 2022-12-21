"""
Helper functions to interface with Pydantic.
"""
from typing import Tuple, Type, Union

import annotated_types
from typing_extensions import Annotated, TypeVar

AnyItemType = TypeVar("AnyItemType")


def contuple(
    item_type: Type[AnyItemType],
    *,
    min_length: Union[int, None] = None,
    max_length: Union[int, None] = None,
) -> Type[Tuple[AnyItemType]]:
    """Like conlist, but a tuple."""
    return Annotated[  # type: ignore[return-value]
        Tuple[item_type],
        annotated_types.Len(min_length or 0, max_length),
    ]
