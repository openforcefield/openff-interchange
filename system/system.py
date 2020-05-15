from pydantic import BaseModel


class System(BaseModel):
    """The OpenFF System object."""

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()
