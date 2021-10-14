"""Models and utilities for processing Foyer data."""
from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Dict, Type

from openff.units import unit
from openff.utilities.utilities import has_package

from openff.interchange.components.mdtraj import (
    _iterate_angles,
    _iterate_propers,
    _OFFBioTop,
    _store_bond_partners,
)
from openff.interchange.components.potentials import Potential, PotentialHandler
from openff.interchange.models import PotentialKey, TopologyKey
from openff.interchange.types import FloatQuantity

if TYPE_CHECKING:
    from foyer.forcefield import Forcefield
    from foyer.topology_graph import TopologyGraph

    from openff.interchange.components.mdtraj import _OFFBioTop

# Is this the safest way to achieve PotentialKey id separation?
POTENTIAL_KEY_SEPARATOR = "-"


if has_package("foyer"):
    from foyer.topology_graph import TopologyGraph  # noqa


def _copy_params(
    params: Dict[str, float], *drop_keys: str, param_units: Dict = None
) -> Dict:
    """Copy parameters from a dictionary."""
    params_copy = copy(params)
    for drop_key in drop_keys:
        params_copy.pop(drop_key, None)
    if param_units:
        for unit_item, units in param_units.items():
            params_copy[unit_item] = params_copy[unit_item] * units
    return params_copy


def _get_potential_key_id(atom_slots: Dict[TopologyKey, PotentialKey], idx):
    """From a dictionary of TopologyKey: PotentialKey, get the PotentialKey id."""
    top_key = TopologyKey(atom_indices=(idx,))
    return atom_slots[top_key].id


def get_handlers_callable() -> Dict[str, Type[PotentialHandler]]:
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


class FoyerVDWHandler(PotentialHandler):
    """Handler storing vdW potentials as produced by a Foyer force field."""

    type: str = "atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    mixing_rule: str = "geometric"
    scale_13: float = 0.0
    scale_14: float = 0.5  # TODO: Replace with Foyer API point?
    scale_15: float = 1.0
    method: str = "cutoff"
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom  # type: ignore

    def store_matches(
        self,
        force_field: "Forcefield",
        topology: "_OFFBioTop",
    ) -> None:
        """Populate self.slot_map with key-val pairs of [TopologyKey, PotentialKey]."""
        from foyer.atomtyper import find_atomtypes

        if topology.n_topology_atoms == 0:
            from parmed.openmm import load_topology

            _store_bond_partners(topology.mdtop)
            top_graph = TopologyGraph.from_parmed(
                load_topology(topology.mdtop.to_openmm())
            )
        else:
            top_graph = TopologyGraph.from_openff_topology(openff_topology=topology)

        if top_graph.number_of_nodes() == 0:
            raise RuntimeError()

        type_map = find_atomtypes(top_graph, forcefield=force_field)
        for key, val in type_map.items():
            top_key = TopologyKey(atom_indices=(key,))
            self.slot_map[top_key] = PotentialKey(id=val["atomtype"])

    def store_potentials(self, force_field: "Forcefield") -> None:
        """Extract specific force field potentials a Forcefield object."""
        for top_key in self.slot_map:
            atom_params = force_field.get_parameters(
                self.type, key=self.slot_map[top_key].id
            )

            atom_params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": unit.kJ / unit.mol, "sigma": unit.nm},
            )

            self.potentials[self.slot_map[top_key]] = Potential(parameters=atom_params)


class FoyerElectrostaticsHandler(PotentialHandler):
    """Handler storing electrostatics potentials as produced by a Foyer force field."""

    type: str = "Electrostatics"
    method: str = "pme"
    expression: str = "coul"
    charges: Dict[TopologyKey, float] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5  # TODO: Replace with Foyer API point?
    scale_15: float = 1.0
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom  # type: ignore

    @property
    def charges_with_virtual_sites(self):
        """Get the total partial charge on each atom, including virtual sites."""
        return self.charges

    def store_charges(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        force_field: "Forcefield",
    ):
        """Look up fixed charges (a.k.a. library charges) from the force field and store them in self.charges."""
        for top_key, pot_key in atom_slots.items():
            foyer_params = force_field.get_parameters("atoms", pot_key.id)
            charge = foyer_params["charge"]
            charge = charge * unit.elementary_charge
            self.charges[top_key] = charge


class FoyerConnectedAtomsHandler(PotentialHandler):
    """Base class for handlers storing valence potentials produced by a Foyer force field."""

    connection_attribute: str = ""
    raise_on_missing_params = True

    def store_matches(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: "_OFFBioTop",
    ) -> None:
        """Populate self.slot_map with key-val pairs of [TopologyKey, PotentialKey]."""
        print(f"Not finding matches with handler type {self.type}")
        """
        for connection in getattr(topology, self.connection_attribute):
            try:
                atoms_iterable = connection.atoms
            except AttributeError:
                atoms_iterable = connection
            atoms_indices = tuple(atom.topology_atom_index for atom in atoms_iterable)
            top_key = TopologyKey(atom_indices=atoms_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atoms_indices
            )

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids)
            )
        """

    def store_potentials(self, force_field: "Forcefield") -> None:
        """Populate self.potentials with key-val pairs of [PotentialKey, Potential]."""
        from foyer.exceptions import MissingForceError, MissingParametersError

        for pot_key in self.slot_map.values():
            try:
                params = force_field.get_parameters(
                    self.type, key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
                )
                params = self.get_params_with_units(params)
                self.potentials[pot_key] = Potential(parameters=params)
            except MissingForceError:
                # Here, we can safely assume that the ForceGenerator is Missing
                self.slot_map = {}
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


class FoyerHarmonicBondHandler(FoyerConnectedAtomsHandler):
    """Handler storing bond potentials as produced by a Foyer force field."""

    type: str = "harmonic_bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    connection_attribute = "topology_bonds"

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            params,
            param_units={"k": unit.kJ / unit.mol / unit.nm ** 2, "length": unit.nm},
        )

    def store_matches(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: "_OFFBioTop",
    ) -> None:
        for bond in topology.mdtop.bonds:
            atoms_indices = tuple((bond.atom1.index, bond.atom2.index))
            top_key = TopologyKey(atom_indices=atoms_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atoms_indices
            )

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids)
            )


class FoyerHarmonicAngleHandler(FoyerConnectedAtomsHandler):
    """Handler storing angle potentials as produced by a Foyer force field."""

    type: str = "harmonic_angles"
    expression: str = "0.5 * k * (theta-angle)**2"
    connection_attribute: str = "angles"

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            {"k": params["k"], "angle": params["theta"]},
            param_units={
                "k": unit.kJ / unit.mol / unit.radian ** 2,
                "angle": unit.dimensionless,
            },
        )

    def store_matches(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: "_OFFBioTop",
    ) -> None:
        for angle in _iterate_angles(topology.mdtop):
            atoms_indices = tuple(a.index for a in angle)
            top_key = TopologyKey(atom_indices=atoms_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atoms_indices
            )

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids)
            )


class FoyerRBProperHandler(FoyerConnectedAtomsHandler):
    """Handler storing Ryckaert-Bellemans proper torsion potentials as produced by a Foyer force field."""

    type: str = "rb_propers"
    expression: str = (
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
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: "_OFFBioTop",
    ) -> None:
        for proper in _iterate_propers(topology.mdtop):
            atoms_indices = tuple(a.index for a in proper)
            top_key = TopologyKey(atom_indices=atoms_indices)

            pot_key_ids = tuple(
                _get_potential_key_id(atom_slots, idx) for idx in atoms_indices
            )

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(pot_key_ids)
            )


class FoyerRBImproperHandler(FoyerRBProperHandler):
    """Handler storing Ryckaert-Bellemans improper torsion potentials as produced by a Foyer force field."""

    type: str = "rb_impropers"
    connection_attribute: str = "impropers"


class FoyerPeriodicProperHandler(FoyerConnectedAtomsHandler):
    """Handler storing periodic proper torsion potentials as produced by a Foyer force field."""

    type: str = "periodic_propers"
    expression: str = "k * (1 + cos(periodicity * phi - phase))"
    connection_attribute: str = "propers"
    raise_on_missing_params: bool = False

    def get_params_with_units(self, params):
        """Get the parameters of this handler, tagged with units."""
        return _copy_params(
            params,
            param_units={
                "k": unit.kJ / unit.mol / unit.nm ** 2,
                "phase": unit.dimensionless,
                "periodicity": unit.dimensionless,
            },
        )


class FoyerPeriodicImproperHandler(FoyerPeriodicProperHandler):
    """Handler storing periodic improper torsion potentials as produced by a Foyer force field."""

    type: str = "periodic_impropers"
    connection_attribute: str = "impropers"


class _RBTorsionHandler(PotentialHandler):
    # TODO: Is this class superceded by FoyerRBProperHandler? Should it be removed?
    type = "Ryckaert-Bellemans"
    expression = (
        "C0 + C1 * (cos(phi - 180)) "
        "C2 * (cos(phi - 180)) ** 2 + C3 * (cos(phi - 180)) ** 3 "
        "C4 * (cos(phi - 180)) ** 4 + C5 * (cos(phi - 180)) ** 5 "
    )
    # independent_variables: Set[str] = {"C0", "C1", "C2", "C3", "C4", "C5"}
