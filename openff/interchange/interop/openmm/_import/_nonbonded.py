from openff.toolkit import Quantity

from openff.interchange.common._nonbonded import ElectrostaticsCollection
from openff.interchange.models import LibraryChargeTopologyKey, TopologyKey


class BasicElectrostaticsCollection(ElectrostaticsCollection):
    """A slightly more complete collection than the base class."""

    _charges: dict[TopologyKey | LibraryChargeTopologyKey, Quantity] = dict()
    _charges_cached: bool = False

    @property
    def charges(
        self,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey, Quantity]:
        """Get the total partial charge on each atom, including virtual sites."""
        if len(self._charges) == 0 or self._charges_cached is False:
            self._charges = self._get_charges()
            self._charges_cached = True

        return self._charges

    def _get_charges(  # type: ignore[override]
        self,
    ) -> dict[TopologyKey | LibraryChargeTopologyKey, Quantity]:
        charges: dict[TopologyKey | LibraryChargeTopologyKey, Quantity] = dict()

        for topology_key, potential_key in self.key_map.items():
            potential = self.potentials[potential_key]

            if len(topology_key.atom_indices) * len(potential.parameters) != 1:
                raise RuntimeError()

            charges[topology_key] = potential.parameters["charge"]

        return charges
