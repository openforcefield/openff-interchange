from abc import abstractmethod
from copy import copy
from typing import Any, Dict, Set

import parmed as pmd
from foyer import Forcefield
from openff.toolkit.topology import Topology

from openff.system import unit as u
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.system import System
from openff.system.models import PotentialKey, TopologyKey

# Is this the safest way to achieve PotentialKey id separation?
POTENTIAL_KEY_SEPARATOR = "-"


def _copy_params(
    params: Dict[str, float], *drop_keys: str, param_units: Dict = None
) -> Dict:
    params_copy = copy(params)
    for drop_key in drop_keys:
        params_copy.pop(drop_key, None)
    if param_units:
        for unit_item, units in param_units.items():
            params_copy[unit_item] = params_copy[unit_item] * units
    return params_copy


def _get_potential_key_id(atom_slots: Dict[TopologyKey, PotentialKey], idx):
    top_key = TopologyKey(atom_indices=(idx,))
    return atom_slots[top_key].id


def _topology_from_parmed(structure: pmd.Structure) -> Topology:
    """ToDo: Create a openFF Topology from a parmed structure"""


def from_foyer(structure: pmd.Structure, ff: Forcefield, **kwargs) -> System:
    """Create an openFF system object from a parmed structure by applying a foyer Forcefield"""
    system = System()
    system.topology = _topology_from_parmed(structure)
    system.handlers["FoyerVDWHandler"] = FoyerAtomTypes()
    system.handlers["FoyerBondHandler"] = FoyerBondHandler()
    system.handlers["FoyerAngleHandler"] = FoyerAngleHandler()
    system.handlers["FoyerPeriodicProperHandler"] = FoyerPeriodicProperHandler()
    system.handlers["FoyerPeriodicImproperHandler"] = FoyerPeriodicImproperHandler()
    system.handlers["FoyerRBProperHandler"] = FoyerRBProperHandler()
    system.handlers["FoyerRBImproperHandler"] = FoyerRBImproperHandler()

    system.handlers["FoyerVDWHandler"].store_matches(forcefield=ff, structure=structure)

    system.handlers["FoyerVDWHandler"].store_potentials(
        forcefield=ff,
    )

    atom_slots = system.handlers["FoyerVDWHandler"].slot_map

    for handler_name, handler in system.handlers.items():
        if handler_name != "FoyerVDWHandler":
            handler.store_matches(structure, atom_slots)

            handler.store_potentials(ff)

    return system


class FoyerAtomTypes(PotentialHandler):
    name: str = "Atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self,
        forcefield: Forcefield,
        structure: pmd.Structure,
    ) -> None:
        """
        Populate self.slot_map with key-val pairs of slots
        and unique potential identifiers

        """
        type_map = forcefield.run_atomtyping(structure)
        for key, val in type_map.items():
            top_key = TopologyKey(atom_indices=(key,))
            self.slot_map[top_key] = PotentialKey(id=val["atomtype"])

    def store_potentials(self, forcefield: Forcefield) -> None:
        for top_key in self.slot_map:
            atom_params = forcefield.get_parameters(
                "atoms", key=[self.slot_map[top_key].id]
            )
            params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": u.kJ / u.mol, "sigma": u.nm},
            )

            self.potentials[self.slot_map[top_key]] = Potential(parameters=params)


class FoyerBondHandler(PotentialHandler):
    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self, structure: pmd.Structure, atom_slots: Dict[TopologyKey, PotentialKey]
    ) -> None:
        for bond in structure.bonds:
            atom_1_idx = bond.atom1.idx
            atom_2_idx = bond.atom2.idx
            top_key = TopologyKey(atom_indices=(atom_1_idx, atom_2_idx))

            atom_1_potential_key_id = _get_potential_key_id(atom_slots, atom_1_idx)
            atom_2_potential_key_id = _get_potential_key_id(atom_slots, atom_2_idx)

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(
                    (atom_1_potential_key_id, atom_2_potential_key_id)
                )
            )

    def store_potentials(self, forcefield: Forcefield) -> None:
        """Store potential for foyer bonds"""
        for _, pot_key in self.slot_map.items():
            bond_params = forcefield.get_parameters(
                "bonds", key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
            )

            bond_params = _copy_params(
                bond_params,
                param_units={"k": u.kJ / u.mol / u.nm ** 2, "length": u.nm},
            )

            self.potentials[pot_key] = Potential(parameters=bond_params)


class FoyerAngleHandler(PotentialHandler):
    name: str = "Angles"
    expression: str = "0.5 * k * (theta-angle)**2"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self, structure: pmd.Structure, atom_slots: Dict[TopologyKey, PotentialKey]
    ) -> None:
        for angle in structure.angles:
            print(angle)
            atom_1_idx = angle.atom1.idx
            atom_2_idx = angle.atom2.idx
            atom_3_idx = angle.atom3.idx
            print(angle)

            top_key = TopologyKey(atom_indices=(atom_1_idx, atom_2_idx, atom_3_idx))

            atom_1_pot_key_id = _get_potential_key_id(atom_slots, atom_1_idx)
            atom_2_pot_key_id = _get_potential_key_id(atom_slots, atom_2_idx)
            atom_3_pot_key_id = _get_potential_key_id(atom_slots, atom_3_idx)

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(
                    (atom_1_pot_key_id, atom_2_pot_key_id, atom_3_pot_key_id)
                )
            )

    def store_potentials(self, forcefield: Forcefield) -> None:
        for _, pot_key in self.slot_map.items():
            angle_params = forcefield.get_parameters(
                "angles", key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
            )

            angle_params = _copy_params(
                {"k": angle_params["k"], "angle": angle_params["theta"]},
                param_units={
                    "k": u.kJ / u.mol / u.nm ** 2,
                    "angle": u.dimensionless,
                },
            )

            self.potentials[pot_key] = Potential(parameters=angle_params)


class FoyerDihedralHandler(PotentialHandler):
    name: str = "PeriodicProper"
    expression: str = "k * (1 + cos(n * phi - phi_eq))"
    independent_variables: Set[str] = {"phi"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    foyer_param_group: str = "periodic_propers"
    parmed_attr: str = "dihedrals"

    def store_matches(
        self, structure: pmd.Structure, atom_slots: Dict[TopologyKey, PotentialKey]
    ) -> None:
        for dihedral in getattr(structure, self.parmed_attr):
            atom_1_idx = dihedral.atom1.idx
            atom_2_idx = dihedral.atom2.idx
            atom_3_idx = dihedral.atom3.idx
            atom_4_idx = dihedral.atom4.idx

            top_key = TopologyKey(
                atom_indices=(atom_1_idx, atom_2_idx, atom_3_idx, atom_4_idx)
            )

            atom_1_pot_key_id = _get_potential_key_id(atom_slots, atom_1_idx)
            atom_2_pot_key_id = _get_potential_key_id(atom_slots, atom_2_idx)
            atom_3_pot_key_id = _get_potential_key_id(atom_slots, atom_3_idx)
            atom_4_pot_key_id = _get_potential_key_id(atom_slots, atom_4_idx)

            self.slot_map[top_key] = PotentialKey(
                id=POTENTIAL_KEY_SEPARATOR.join(
                    (
                        atom_1_pot_key_id,
                        atom_2_pot_key_id,
                        atom_3_pot_key_id,
                        atom_4_pot_key_id,
                    )
                )
            )

    def store_potentials(self, forcefield: Forcefield) -> None:
        for _, pot_key in self.slot_map.items():
            foyer_params = forcefield.get_parameters(
                self.foyer_param_group, key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
            )

            foyer_params = self.assign_units_to_params(foyer_params)

            self.potentials[pot_key] = Potential(parameters=foyer_params)

    @abstractmethod
    def assign_units_to_params(self, foyer_params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FoyerPeriodicProperHandler(FoyerDihedralHandler):
    name: str = "PeriodicProper"
    expression: str = "k * (1 + cos(periodicity * phi - phase))"
    independent_variables: Set[str] = {"phi"}
    foyer_param_group: str = "periodic_propers"
    parmed_attr: str = "dihedrals"

    def assign_units_to_params(self, foyer_params: Dict[str, Any]) -> Dict[str, Any]:
        periodic_params = _copy_params(
            {
                "k": foyer_params["k"],
                "periodicity": foyer_params["periodicity"],
                "phase": foyer_params["phase"],
            },
            param_units={
                "k": u.kJ / u.mol / u.nm ** 2,
                "phase": u.dimensionless,
                "periodicity": u.dimensionless,
            },
        )
        return periodic_params


class FoyerPeriodicImproperHandler(FoyerPeriodicProperHandler):
    name: str = "PeriodicImproper"
    foyer_param_group: str = "periodic_impropers"
    parmed_attr: str = "impropers"


class FoyerRBProperHandler(FoyerDihedralHandler):
    name: str = "RBPropers"
    expression: str = (
        "c0 * cos(phi)**0 + c1 * cos(phi)**1 + "
        "c2 * cos(phi)**2 + c3 * cos(phi)**3 + "
        "c4 * cos(phi)**4 + c5 * cos(phi)**5"
    )
    independent_variables: Set[str] = {"phi"}
    foyer_param_group: str = "rb_propers"
    parmed_attr: str = "dihedrals"

    def assign_units_to_params(self, foyer_params: Dict[str, Any]) -> Dict[str, Any]:
        rb_params = _copy_params(
            foyer_params,
            param_units={
                "c0": u.KJ / u.mol,
                "c1": u.KJ / u.mol,
                "c2": u.KJ / u.mol,
                "c3": u.KJ / u.mol,
                "c4": u.KJ / u.mol,
                "c5": u.KJ / u.mol,
            },
        )

        return rb_params


class FoyerRBImproperHandler(FoyerRBProperHandler):
    name: str = "RBImpropers"
    foyer_param_group: str = "rb_impropers"
    parmed_attr: str = "impropers"
