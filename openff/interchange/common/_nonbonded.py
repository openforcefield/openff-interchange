import abc
from collections.abc import Iterable
from typing import Any, Literal

from openff.toolkit import Quantity, unit
from pydantic import Field, PrivateAttr, computed_field

from openff.interchange._annotations import _DistanceQuantity, _ElementaryChargeQuantity
from openff.interchange.components.potentials import Collection
from openff.interchange.models import (
    LibraryChargeTopologyKey,
    TopologyKey,
    VirtualSiteKey,
)


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

    expression: Literal["4*epsilon*((sigma/r)**12-(sigma/r)**6)"] = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"

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
        "reaction-field",
    ] = Field("Ewald3D-ConductingBoundary")
    nonperiodic_potential: Literal[
        "Coulomb",
        "cutoff",
        "no-cutoff",
        "reaction-field",
    ] = Field("Coulomb")
    exception_potential: Literal["Coulomb"] = Field("Coulomb")

    _charges: dict[Any, _ElementaryChargeQuantity] = PrivateAttr()
    _charges_cached: bool = PrivateAttr(default=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def charges(
        self,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey | VirtualSiteKey, _ElementaryChargeQuantity]:
        """Get the total partial charge on each atom, including virtual sites."""
        if len(self._charges) == 0 or self._charges_cached is False:
            self._charges.update(self._get_charges(include_virtual_sites=False))
            self._charges_cached = True

        return self._charges

    def get_charge_array(self, include_virtual_sites: bool = False) -> Quantity:
        """
        Return a one-dimensional array-like of atomic charges, ordered topologically.

        If virtual sites are present in the system, `NotImplementedError` is raised.
        """
        if include_virtual_sites:
            raise NotImplementedError("Not yet implemented with virtual sites")

        if VirtualSiteKey in {type(key) for key in self.key_map}:
            raise NotImplementedError(
                "Not yet implemented when virtual sites are present, even with `include_virtual_sites=False`.",
            )

        return Quantity.from_list([q for _, q in sorted(self.charges.items(), key=lambda x: x[0].atom_indices)])

    def _get_charges(
        self,
        include_virtual_sites: bool = False,
    ) -> dict[TopologyKey | VirtualSiteKey | LibraryChargeTopologyKey, _ElementaryChargeQuantity]:
        if include_virtual_sites:
            raise NotImplementedError()

        return {
            topology_key: self.potentials[potential_key].parameters["charge"]
            for topology_key, potential_key in self.key_map.items()
        }
