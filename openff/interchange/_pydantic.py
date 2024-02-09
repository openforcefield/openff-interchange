try:
    from pydantic.v1 import (
        Field,
        PositiveInt,
        PrivateAttr,
        ValidationError,
        conint,
        validator,
    )
except ImportError:
    from pydantic import (  # type: ignore[assignment]
        Field,
        PositiveInt,
        PrivateAttr,
        ValidationError,
        conint,
        validator,
    )
