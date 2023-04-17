"""Models and utilities for processing Foyer data."""
from abc import abstractmethod
from copy import copy
from typing import Optional

from foyer.forcefield import Forcefield
from openff.models.types import FloatQuantity
from openff.toolkit.topology import Topology
from openff.units import Quantity, unit
from pydantic import PrivateAttr

from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ProperTorsionCollection,
    RyckaertBellemansTorsionCollection,
)
from openff.interchange.components.potentials import Collection, Potential
from openff.interchange.constants import _PME
from openff.interchange.models import PotentialKey, TopologyKey

# Is this the safest way to achieve PotentialKey id separation?
POTENTIAL_KEY_SEPARATOR = "-"


_CollectionAlias = type[Collection]


def _copy_params(
    params: dict[str, float],
    *drop_keys: str,
    param_units: Optional[dict] = None,
) -> dict:
    """Copy parameters from a dictionary."""
    params_copy = copy(params)
    for drop_key in drop_keys:
        params_copy.pop(drop_key, None)
    if param_units:
        for unit_item, units in param_units.items():
            params_copy[unit_item] = params_copy[unit_item] * units
    return params_copy


def _get_potential_key_id(atom_slots: dict[TopologyKey, PotentialKey], idx):
    """From a dictionary of TopologyKey: PotentialKey, get the PotentialKey id."""
    top_key = TopologyKey(atom_indices=(idx,))
    return atom_slots[top_key].id


def get_handlers_callable() -> dict[str, _CollectionAlias]:
    """Map Foyer-style handlers from string identifiers."""
    return {
        "vdW": FoyerVDWHandler,
        "Electrostatics": FoyerElectrostaticsHandler,
        "Bonds": FoyerHarmonicBondHandler,
        "Angles": FoyerHarmonicAngleHandler,
        "RBTorsions": FoyerRBProperHandler,
        "RBImpropers": FoyerRBImproperHandler,
        "ProperTorsions": FoyerPeriodicProperHandler,
        "ImproperTorsions": FoyerPeriodicImproperHandler,
    }


class FoyerVDWHandler(vdWCollection):
    """Handler storing vdW potentials as produced by a Foyer force field."""

    force_field_key: str = "atoms"
    mixing_rule = "geometric"
    method: str = "cutoff"
    switch_width: FloatQuantity["angstrom"] = 0.0 * unit.angstrom

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
    periodic_potential: str = _PME
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom

    _charges: dict[TopologyKey, Quantity] = PrivateAttr(dict())

    @property
    def charges(self) -> dict[TopologyKey, Quantity]:
        """Get the total partial charge on each atom, including virtual sites."""
        return self._charges

    @property
    def charges_with_virtual_sites(self):
        """Get the total partial charge on each atom, including virtual sites."""
        raise NotImplementedError("Foyer force fields do not support virtual sites.")

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


class FoyerConnectedAtomsHandler(Collection):
    """Base class for handlers storing valence potentials produced by a Foyer force field."""

    force_field_key: str
    connection_attribute: str = ""
    raise_on_missing_params: bool = True

    def store_matches(
        self,
        atom_slots: dict[TopologyKey, PotentialKey],
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        for connection in getattr(topology, self.connection_attribute):
            try:
                atoms_iterable = connection.atoms
            except AttributeError:
                atoms_iterable = connection
            atom_indices = tuple(topology.atom_index(atom) for atom in atoms_iterable)

            top_key = TopologyKey(atom_indices=atom_indices)
            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atom_indices
            )

            self.key_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids),
            )

    def store_potentials(self, force_field: "Forcefield") -> None:
        """Populate self.potentials with key-val pairs of [PotentialKey, Potential]."""
        from foyer.exceptions import MissingForceError, MissingParametersError

        for pot_key in self.key_map.values():
            try:
                params = force_field.get_parameters(
                    self.force_field_key,
                    key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR),
                )
                params = self.get_params_with_units(params)
                self.potentials[pot_key] = Potential(parameters=params)
            except MissingForceError:
                # Here, we can safely assume that the ForceGenerator is Missing
                self.key_map = {}
                self.potentials = {}
                return
            except MissingParametersError as e:
                if self.raise_on_missing_params:
                    raise e
                else:
                    pass

    @abstractmethod
    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        raise NotImplementedError


class FoyerHarmonicBondHandler(FoyerConnectedAtomsHandler, BondCollection):
    """Handler storing bond potentials as produced by a Foyer force field."""

    type = "Bonds"
    expression = "k/2*(r-length)**2"
    force_field_key = "harmonic_bonds"
    connection_attribute = "bonds"

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            params,
            param_units={"k": unit.kJ / unit.mol / unit.nm**2, "length": unit.nm},
        )

    def store_matches(
        self,
        atom_slots: dict[TopologyKey, PotentialKey],
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        for bond in topology.bonds:
            atom_indices = (
                topology.atom_index(bond.atom1),
                topology.atom_index(bond.atom2),
            )
            top_key = TopologyKey(atom_indices=atom_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atom_indices
            )

            self.key_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids),
            )


class FoyerHarmonicAngleHandler(FoyerConnectedAtomsHandler, AngleCollection):
    """Handler storing angle potentials as produced by a Foyer force field."""

    type = "Angles"
    expression = "k/2*(theta-angle)**2"
    force_field_key: str = "harmonic_angles"
    connection_attribute: str = "angles"

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            {"k": params["k"], "angle": params["theta"]},
            param_units={
                "k": unit.kJ / unit.mol / unit.radian**2,
                "angle": unit.dimensionless,
            },
        )

    def store_matches(
        self,
        atom_slots: dict[TopologyKey, PotentialKey],
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        for angle in topology.angles:
            atom_indices = tuple(topology.atom_index(atom) for atom in angle)
            top_key = TopologyKey(atom_indices=atom_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atom_indices
            )

            self.key_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids),
            )


class FoyerRBProperHandler(
    FoyerConnectedAtomsHandler,
    RyckaertBellemansTorsionCollection,
):
    """Handler storing Ryckaert-Bellemans proper torsion potentials as produced by a Foyer force field."""

    force_field_key = "rb_propers"
    type = "RBTorsions"
    expression = (
        "C0 * cos(phi)**0 + C1 * cos(phi)**1 + "
        "C2 * cos(phi)**2 + C3 * cos(phi)**3 + "
        "C4 * cos(phi)**4 + C5 * cos(phi)**5"
    )
    connection_attribute: str = "propers"
    raise_on_missing_params: bool = False

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        rb_params = {k.upper(): v for k, v in params.items()}
        param_units = {k: unit.kJ / unit.mol for k in rb_params}
        return _copy_params(rb_params, param_units=param_units)

    def store_matches(
        self,
        atom_slots: dict[TopologyKey, PotentialKey],
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        for proper in topology.propers:
            atom_indices = tuple(topology.atom_index(atom) for atom in proper)
            top_key = TopologyKey(atom_indices=atom_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atom_indices
            )

            self.key_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids),
            )


class FoyerRBImproperHandler(FoyerRBProperHandler):
    """Handler storing Ryckaert-Bellemans improper torsion potentials as produced by a Foyer force field."""

    type = "RBImpropers"  # type: ignore[assignment]
    connection_attribute: str = "impropers"


class FoyerPeriodicProperHandler(FoyerConnectedAtomsHandler, ProperTorsionCollection):
    """Handler storing periodic proper torsion potentials as produced by a Foyer force field."""

    force_field_key = "periodic_propers"
    connection_attribute: str = "propers"
    raise_on_missing_params: bool = False
    type = "ProperTorsions"
    expression = "k*(1+cos(periodicity*theta-phase))"

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            params,
            param_units={
                "k": unit.kJ / unit.mol / unit.nm**2,
                "phase": unit.dimensionless,
                "periodicity": unit.dimensionless,
            },
        )


class FoyerPeriodicImproperHandler(FoyerPeriodicProperHandler):
    """Handler storing periodic improper torsion potentials as produced by a Foyer force field."""

    type = "ImproperTorsions"  # type: ignore[assignment]
    connection_attribute: str = "impropers"


class _RBTorsionHandler(RyckaertBellemansTorsionCollection):
    pass
