from typing import Literal

from openff.toolkit import Quantity, Topology, unit
from pydantic import Field, PrivateAttr

from openff.interchange._annotations import _DistanceQuantity
from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.components.potentials import Potential
from openff.interchange.foyer._base import _copy_params
from openff.interchange.foyer._guard import has_foyer
from openff.interchange.models import PotentialKey, TopologyKey

if has_foyer:
    try:
        from foyer import Forcefield
    except ModuleNotFoundError:
        pass


class FoyerVDWHandler(vdWCollection):
    """Handler storing vdW potentials as produced by a Foyer force field."""

    force_field_key: str = "atoms"
    mixing_rule: Literal["lorentz-berthelot", "geometric"] = Field("geometric")
    method: Literal["cutoff", "pme", "no-cutoff"] = Field("cutoff")

    def store_matches(
        self,
        force_field: "Forcefield",
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        from foyer.atomtyper import find_atomtypes
        from foyer.topology_graph import TopologyGraph

        top_graph = TopologyGraph.from_openff_topology(topology)

        type_map = find_atomtypes(top_graph, forcefield=force_field)
        for key, val in type_map.items():
            top_key = TopologyKey(atom_indices=(key,))
            self.key_map[top_key] = PotentialKey(id=val["atomtype"])

    def store_potentials(self, force_field: "Forcefield") -> None:
        """Extract specific force field potentials a Forcefield object."""
        for top_key in self.key_map:
            atom_params = force_field.get_parameters(
                self.force_field_key,
                key=self.key_map[top_key].id,
            )

            atom_params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": unit.kJ / unit.mol, "sigma": unit.nm},
            )

            self.potentials[self.key_map[top_key]] = Potential(parameters=atom_params)


class FoyerElectrostaticsHandler(ElectrostaticsCollection):
    """Handler storing electrostatics potentials as produced by a Foyer force field."""

    force_field_key: str = "atoms"
    cutoff: _DistanceQuantity = 9.0 * unit.angstrom

    _charges: dict[TopologyKey, Quantity] = PrivateAttr(dict())

    def store_charges(
        self,
        atom_slots: dict[TopologyKey, PotentialKey],
        force_field: "Forcefield",
    ):
        """Look up fixed charges (a.k.a. library charges) from the force field and store them in self._charges."""
        for top_key, pot_key in atom_slots.items():
            foyer_params = force_field.get_parameters(self.force_field_key, pot_key.id)

            charge = Quantity(foyer_params["charge"], unit.elementary_charge)

            self._charges[top_key] = charge

            self.key_map[top_key] = pot_key
            self.potentials[pot_key] = Potential(
                parameters={"charge": charge},
            )
