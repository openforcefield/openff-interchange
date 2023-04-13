import abc
from collections import defaultdict
from typing import DefaultDict, Dict, Literal

from openff.models.types import FloatQuantity
from openff.units import Quantity, unit
from pydantic import Field

from openff.interchange.components.potentials import Collection
from openff.interchange.models import TopologyKey


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


class vdWCollection(_NonbondedCollection):
    """Handler storing vdW potentials."""

    type: Literal["vdW"] = "vdW"

    expression: Literal[
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

    mixing_rule: Literal["lorentz-berthelot"] = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule) used in computing pairwise vdW interactions",
    )


class ElectrostaticsCollection(_NonbondedCollection):
    """Handler storing electrostatics interactions."""

    type: Literal["Electrostatics"] = "Electrostatics"

    expression: Literal["coul"] = "coul"

    @property
    def charges(self) -> Dict[TopologyKey, Quantity]:
        """Get the total partial charge on each atom, excluding virtual sites."""
        return self.get_charges(include_virtual_sites=False)

    def get_charges(
        self,
        include_virtual_sites: bool = False,
    ) -> Dict[TopologyKey, Quantity]:
        """Get the total partial charge on each atom or particle."""
        if include_virtual_sites:
            raise NotImplementedError()

        charges: DefaultDict[TopologyKey, Quantity] = defaultdict(
            lambda: Quantity(0.0, unit.elementary_charge),
        )

        for topology_key, potential_key in self.key_map.items():
            potential = self.potentials[potential_key]

            charges[topology_key] = potential.parameters["charge"]

        return charges
