from typing import Optional, Sequence

from pydantic import BaseModel, Field

from openff.system import unit
from openff.system.types import custom_quantity_encoder, json_loader


class DefaultModel(BaseModel):
    class Config:
        json_encoders = {
            unit.Quantity: custom_quantity_encoder,
        }
        json_loads = json_loader
        validate_assignment = True
        arbitrary_types_allowed = True


class TopologyKey(DefaultModel):
    # Should be Tuple[int], see issues #495
    atom_indices: Sequence[int] = Field(
        tuple(), description="The indices of the atoms occupied by this interaction"
    )
    mult: Optional[int] = Field(
        None, description="The index of this duplicate interaction"
    )

    def __hash__(self):
        return hash((self.atom_indices, self.mult))


class PotentialKey(DefaultModel):
    id: str = Field(
        ...,
        description="A unique identifier of this potential, i.e. a SMARTS pattern or an atom type",
    )
    mult: Optional[int] = Field(
        None, description="The index of this duplicate interaction"
    )

    def __hash__(self):
        return hash((self.id, self.mult))
