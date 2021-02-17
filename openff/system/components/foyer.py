from copy import copy
from typing import Dict, Set, Any, Tuple


import parmed as pmd

from foyer import Forcefield

from openff.toolkit.topology import Topology
from openff.system import unit as u
from openff.system.components.system import System
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.exceptions import MissingParametersError


def _copy_params(params: Dict[str, float],
                 *drop_keys: str,
                 param_units: Dict = None) -> Dict:
    params_copy = copy(params)
    for drop_key in drop_keys:
        params_copy.pop(drop_key, None)
    if param_units:
        for unit_item, units in param_units.items():
            params_copy[unit_item] = params_copy[unit_item] * units
    return params_copy


def _get_openmm_force_gen(ff: Forcefield, gen_type: type) -> Any:
    return list(
            filter(
                lambda x: isinstance(x, gen_type),
                ff.getGenerators()
            )
        ).pop()


def _topology_from_parmed(structure: pmd.Structure) -> Topology:
    """ToDo: Create a openFF Topology from a parmed structure"""


def from_foyer(structure: pmd.Structure,
               ff: Forcefield,
               **kwargs) -> System:
    """Create an openFF system object from a parmed structure by applying a foyer Forcefield"""
    system = System()
    system.topology = _topology_from_parmed(structure)
    foyer_atom_handler = FoyerAtomTypes()
    foyer_bond_handler = FoyerBondHandler()
    system.handlers['FoyerAtomHandler'] = foyer_atom_handler
    system.handlers['FoyerBondHandler'] = foyer_bond_handler

    return system


class FoyerAtomTypes(PotentialHandler):

    name: str = "Atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[int, str] = dict()
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

    def store_potentials(self, forcefield) -> None:
        from simtk.openmm.app.forcefield import NonbondedGenerator
        non_bonded_forces_gen = _get_openmm_force_gen(forcefield, NonbondedGenerator)

        if non_bonded_forces_gen:
            nonbonded_params = non_bonded_forces_gen.params.paramsForType

            for atom_idx in self.slot_map:
                try:
                    params = _copy_params(
                        nonbonded_params[self.slot_map[atom_idx]],
                        'charge',
                        param_units={
                            'epsilon': u.kcal / u.mol,
                            'sigma': u.nm
                        })
                except KeyError:
                    raise MissingParametersError(
                        f'Missing parameters for atomtype {self.slot_map[atom_idx]}'
                        'in the forcefield'
                    )

                self.potentials[atom_idx] = Potential(
                    parameters=params
                )
        else:
            raise MissingParametersError(
                'The forcefield is missing NonBondedForces Parameters'
            )


class FoyerBondHandler(PotentialHandler):

    name: str = "Bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[Tuple[int, int], Tuple[str, str]] = dict()
    potentials: Dict[Tuple[int, int], Potential] = dict()

    def store_matches(self,
                      structure: pmd.Structure,
                      atom_slots: Dict[str, str]) -> None:
        for bond in structure.bonds:
            atom_1_idx = bond.atom1.idx
            atom_2_idx = bond.atom2.idx

            self.slot_map[(atom_1_idx, atom_2_idx)] = \
                (atom_slots[atom_1_idx], atom_slots[atom_2_idx])

    def store_potentials(self, forcefield: Forcefield) -> None:
        from simtk.openmm.app.forcefield import HarmonicBondGenerator
        harmonic_bond_forces_gen = _get_openmm_force_gen(forcefield, HarmonicBondGenerator)
        bonds_for_atom_types = harmonic_bond_forces_gen.bondsForAtomType

        for (atom_1_idx, atom_2_idx), (atom_1_type, atom_2_type) in self.slot_map.items():
            for i in bonds_for_atom_types[atom_1_type]:
                types1 = harmonic_bond_forces_gen.types1[i]
                types2 = harmonic_bond_forces_gen.types2[i]

                # Replicated from
                # https://github.com/openmm/openmm/blob/b49b82efb5a253a7c891ca084b3370e181de2ea3/wrappers/python/simtk/openmm/app/forcefield.py#L1960-L1983

                if (atom_1_type in types1 and atom_2_type in types2) \
                        or (atom_1_type in types2 and atom_2_type in types1):
                    params = {
                        'k': harmonic_bond_forces_gen.k[i] * u.kcal / u.mol / u.nm**2,
                        'length': harmonic_bond_forces_gen.length[i] * u.nm
                    }

                    self.potentials[(atom_1_idx, atom_2_idx)] = Potential(parameters=params)
                    break

            if not self.potentials.get((atom_1_idx, atom_2_idx)):
                raise MissingParametersError(
                    f'Could not find parameters for the Bond between '
                    f'atoms {atom_1_type}-{atom_2_type}'
                )


        # for identifier in self.slot_map.values():
        #     # look up FF data from identifier (or directly consult the
        #     # topology, which would involve droppint the make_slot_map method)
        #     # store FF data into a Potential object
        #     self.potentials[identifier] = Potential(...)
