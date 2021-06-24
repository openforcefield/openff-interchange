from typing import Literal, Optional, Tuple

from openff.units import unit
from pydantic import BaseModel, Field

from openff.interchange.types import custom_quantity_encoder, json_loader


class DefaultModel(BaseModel):
    class Config:
        json_encoders = {
            unit.Quantity: custom_quantity_encoder,
        }
        json_loads = json_loader
        validate_assignment = True
        arbitrary_types_allowed = True


class TopologyKey(DefaultModel):
    atom_indices: Tuple[int, ...] = Field(
        tuple(), description="The indices of the atoms occupied by this interaction"
    )
    mult: Optional[int] = Field(
        None, description="The index of this duplicate interaction"
    )

    def __hash__(self):
        return hash((self.atom_indices, self.mult))


class VirtualSiteKey(DefaultModel):
    atom_indices: Tuple[int, ...] = Field(
        tuple(), description="The indices of the atoms that anchor this virtual site"
    )
    type: str = Field(description="The type of this virtual site")
    match: Literal["once", "all_permutations"] = Field(
        "The `match` attribute of the associated virtual site type"
    )

    def __hash__(self):
        return hash((self.atom_indices, self.type))


class PotentialKey(DefaultModel):
    id: str = Field(
        ...,
        description="A unique identifier of this potential, i.e. a SMARTS pattern or an atom type",
    )
    mult: Optional[int] = Field(
        None, description="The index of this duplicate interaction"
    )
    associated_handler: Optional[str] = Field(
        None,
        description="The type of handler this potential key is associated with, "
        "i.e. 'Bonds', 'vdW', or 'LibraryCharges",
    )

    def __hash__(self):
        return hash((self.id, self.mult))
