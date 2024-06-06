import abc
from collections.abc import Iterable
from typing import Literal

from openff.toolkit import Quantity, unit

from openff.interchange._annotations import _DistanceQuantity
from openff.interchange._pydantic import Field, PrivateAttr
from openff.interchange.components.potentials import Collection
from openff.interchange.constants import _PME
from openff.interchange.models import LibraryChargeTopologyKey, TopologyKey


class _NonbondedCollection(Collection, abc.ABC):
    type: str = "nonbonded"

    cutoff: _DistanceQuantity = Field(
        Quantity(10.0, unit.angstrom),
        description="The distance at which pairwise interactions are truncated",
    )

    scale_12: float = Field(
        0.0,
        description="The scaling factor applied to 1-2 interactions",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Hack to get electrostatics subclasses to have private attributes."""
        if "Electrostatics" in cls.__name__:
            cls._charges = dict()
            cls._charges_cached = False

        return super().__pydantic_init_subclass__(**kwargs)


class vdWCollection(_NonbondedCollection):
    """Handler storing vdW potentials."""

    type: Literal["vdW"] = "vdW"

    expression: Literal["4*epsilon*((sigma/r)**12-(sigma/r)**6)"] = (
        "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    )

    mixing_rule: str = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule) used in computing pairwise vdW interactions",
    )

    switch_width: _DistanceQuantity = Field(
        Quantity(1.0, unit.angstrom),
        description="The width over which the switching function is applied",
    )

    periodic_method: Literal["cutoff", "no-cutoff", "pme"] = Field("cutoff")
    nonperiodic_method: Literal["cutoff", "no-cutoff", "pme"] = Field("no-cutoff")

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 1.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a list of names of parameters included in each potential in this colletion."""
        return "sigma", "epsilon"


class ElectrostaticsCollection(_NonbondedCollection):
    """Handler storing electrostatics interactions."""

    type: Literal["Electrostatics"] = "Electrostatics"

    expression: Literal["coul"] = "coul"

    periodic_potential: Literal[
        "Ewald3D-ConductingBoundary",
        "cutoff",
        "no-cutoff",
    ] = Field(_PME)
    nonperiodic_potential: Literal["Coulomb", "cutoff", "no-cutoff"] = Field("Coulomb")
    exception_potential: Literal["Coulomb"] = Field("Coulomb")

    _charges: dict[
        TopologyKey | LibraryChargeTopologyKey,
        Quantity,
    ] = PrivateAttr(
        default_factory=dict,
    )
    _charges_cached: bool = PrivateAttr(default=False)

    @property
    def charges(self) -> dict[TopologyKey, Quantity]:
        """Get the total partial charge on each atom, including virtual sites."""
        if len(self._charges) == 0 or self._charges_cached is False:
            self._charges = self._get_charges(include_virtual_sites=False)
            self._charges_cached = True

        return self._charges

    def _get_charges(
        self,
        include_virtual_sites: bool = False,
    ) -> dict[TopologyKey, Quantity]:
        if include_virtual_sites:
            raise NotImplementedError()

        return {
            topology_key: self.potentials[potential_key].parameters["charge"]
            for topology_key, potential_key in self.key_map.items()
        }
