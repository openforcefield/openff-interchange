from copy import copy
from typing import Dict, Set, Tuple

import parmed as pmd
from foyer import Forcefield
from openff.toolkit.topology import Topology

from openff.system import unit as u
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.system import System


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


def _topology_from_parmed(structure: pmd.Structure) -> Topology:
    """ToDo: Create a openFF Topology from a parmed structure"""


def from_foyer(structure: pmd.Structure, ff: Forcefield, **kwargs) -> System:
    """Create an openFF system object from a parmed structure by applying a foyer Forcefield"""
    system = System()
    system.topology = _topology_from_parmed(structure)
    foyer_atom_handler = FoyerAtomTypes()
    foyer_bond_handler = FoyerBondHandler()
    system.handlers["FoyerAtomHandler"] = foyer_atom_handler
    system.handlers["FoyerBondHandler"] = foyer_bond_handler

    return system


class FoyerAtomTypes(PotentialHandler):

    name: str = "Atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[int, str] = dict()  # type: ignore
    potentials: Dict[str, Potential] = dict()

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
            self.slot_map[key] = val["atomtype"]

    def store_potentials(self, forcefield: Forcefield) -> None:
        for atom_idx in self.slot_map:
            atom_params = forcefield.get_parameters(
                'atoms',
                key=[self.slot_map[atom_idx]]
            )
            params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": u.kcal / u.mol, "sigma": u.nm},
            )

            self.potentials[self.slot_map[atom_idx]] = Potential(parameters=params)


class FoyerBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[Tuple[int, int], Tuple[str, str]] = dict()  # type: ignore
    potentials: Dict[Tuple[int, int], Potential] = dict()  # type: ignore

    def store_matches(
        self, structure: pmd.Structure, atom_slots: Dict[str, str]
    ) -> None:
        for bond in structure.bonds:
            atom_1_idx = bond.atom1.idx
            atom_2_idx = bond.atom2.idx

            self.slot_map[(atom_1_idx, atom_2_idx)] = (
                atom_slots[atom_1_idx],
                atom_slots[atom_2_idx],
            )

    def store_potentials(self, forcefield: Forcefield) -> None:
        """Store potential for foyer bonds"""
        for (atom_1_idx, atom_2_idx), (
            atom_1_type,
            atom_2_type,
        ) in self.slot_map.items():
            bond_params = forcefield.get_parameters(
                'bonds',
                key=[atom_1_type, atom_2_type]
            )

            bond_params = _copy_params(
                bond_params,
                param_units={
                    'k': u.kcal / u.mol / u.nm ** 2,
                    'length': u.nm
                }
            )

            self.potentials[(atom_1_idx, atom_2_idx)] = Potential(
                parameters=bond_params
            )


class FoyerAngleHandler(PotentialHandler):
    name: str = "Angle"
    expression: str = "0.5 * k * (theta-theta_eq)**2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[Tuple[int, int, int], Tuple[str, str, str]] = dict()  # type: ignore
    potentials: Dict[Tuple[int, int, int], Potential] = dict()  # type: ignore

    def store_matches(
        self, structure: pmd.Structure, atom_slots: Dict[str, str]
    ) -> None:
        for bond in structure.angles:
            atom_1_idx = bond.atom1.idx
            atom_2_idx = bond.atom2.idx
            atom_3_idx = bond.atom3.idx

            self.slot_map[(atom_1_idx, atom_2_idx, atom_3_idx)] = (
                atom_slots[atom_1_idx],
                atom_slots[atom_2_idx],
                atom_slots[atom_3_idx]
            )

    def store_potentials(self, forcefield: Forcefield) -> None:
        for (atom_1_idx, atom_2_idx, atom_3_idx), (
            atom_1_type,
            atom_2_type,
            atom_3_type
        ) in self.slot_map.items():
            angle_params = forcefield.get_parameters(
                'angles',
                key=[atom_1_type, atom_2_type, atom_3_type]
            )

            angle_params = _copy_params(
                {'k': angle_params['k'], 'theta_eq': angle_params['theta']},
                param_units={
                    'k': u.kcal / u.mol / u.nm ** 2,
                    'theta': u.dimensionless
                }
            )

            self.potentials[(atom_1_idx, atom_2_idx, atom_3_idx)] = Potential(
                parameters=angle_params
            )
