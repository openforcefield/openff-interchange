import abc
from typing import Literal

from openff.models.types import FloatQuantity
from openff.units import unit
from pydantic import Field

from openff.interchange.components.potentials import Collection


class _NonbondedCollection(Collection, abc.ABC):  # noqa
    type: str = "nonbonded"

    cutoff: FloatQuantity["angstrom"] = Field(  # noqa
        unit.Quantity(9.0, unit.angstrom),
        description="The distance at which pairwise interactions are truncated",
    )

    scale_13: float = Field(
        0.0,
        description="The scaling factor applied to 1-3 interactions",
    )
    scale_14: float = Field(
        0.5,
        description="The scaling factor applied to 1-4 interactions",
    )
    scale_15: float = Field(
        1.0,
        description="The scaling factor applied to 1-5 interactions",
    )


class _vdWCollection(_NonbondedCollection):
    """Handler storing vdW potentials."""

    type: Literal["vdW"] = "vdW"

    expression: Literal[
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

    mixing_rule: Literal["lorentz-berthelot"] = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule) used in computing pairwise vdW interactions",
    )


class _ElectrostaticsCollection(_NonbondedCollection):
    """Handler storing electrostatics interactions."""

    type: Literal["Electrostatics"] = "Electrostatics"

    expression: Literal["coul"] = "coul"
