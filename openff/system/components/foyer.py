from abc import abstractmethod
from copy import copy
from typing import Dict, Set, Type

from ele import element_from_atomic_number
from foyer import Forcefield
from foyer.exceptions import MissingForceError, MissingParametersError
from foyer.topology_graph import TopologyGraph
from openff.toolkit.topology import Topology

from openff.system import unit as u
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.system import System
from openff.system.models import PotentialKey, TopologyKey

# Is this the safest way to achieve PotentialKey id separation?
POTENTIAL_KEY_SEPARATOR = "-"


@classmethod  # type: ignore
def from_off_topology(cls, off_topology: Topology) -> TopologyGraph:
    top_graph = cls()
    for top_atom in off_topology.topology_atoms:
        atom = top_atom.atom
        element = element_from_atomic_number(atom.atomic_number)
        top_graph.add_atom(
            name=atom.name,
            index=top_atom.topology_atom_index,
            atomic_number=element.atomic_number,
            element=element.symbol,
        )
    for top_bond in off_topology.topology_bonds:
        atoms_indices = [atom.topology_atom_index for atom in top_bond.atoms]
        top_graph.add_bond(atoms_indices[0], atoms_indices[1])
    return top_graph


TopologyGraph.from_off_topology = from_off_topology


def _copy_params(
    params: Dict[str, float], *drop_keys: str, param_units: Dict = None
) -> Dict:
    """copy parameters from a dictionary"""
    params_copy = copy(params)
    for drop_key in drop_keys:
        params_copy.pop(drop_key, None)
    if param_units:
        for unit_item, units in param_units.items():
            params_copy[unit_item] = params_copy[unit_item] * units
    return params_copy


def _get_potential_key_id(atom_slots: Dict[TopologyKey, PotentialKey], idx):
    """From a dictionary of TopologyKey: PotentialKey, get the PotentialKey id"""
    top_key = TopologyKey(atom_indices=(idx,))
    return atom_slots[top_key].id


def from_foyer(topology: Topology, ff: Forcefield, **kwargs) -> System:
    system = System()
    system.topology = topology

    for name, Handler in get_handlers_callable().items():
        system.handlers[name] = Handler()

    system.handlers["vdw"].store_matches(ff, topology=topology)
    system.handlers["vdw"].store_potentials(forcefield=ff)  # type: ignore

    atom_slots = system.handlers["vdw"].slot_map
    for name, handler in system.handlers.items():
        if name != "vdw":
            handler.store_matches(atom_slots, topology=topology)
            handler.store_potentials(ff)

    return system


def get_handlers_callable() -> Dict[str, Type[PotentialHandler]]:
    return {
        "vdw": FoyerVDWHandler,
        "harmonic_bonds": FoyerHarmonicBondHandler,
        "harmonic_angles": FoyerHarmonicAngleHandler,
        "rb_propers": FoyerRBProperHandler,
        "rb_impropers": FoyerRBImproperHandler,
        "periodic_propers": FoyerPeriodicProperHandler,
        "periodic_impropers": FoyerPeriodicImproperHandler,
    }


class FoyerVDWHandler(PotentialHandler):
    name: str = "atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()

    def store_matches(
        self,
        forcefield: Forcefield,
        topology: Topology,
    ) -> None:
        """Populate slotmap with key-val pairs of slots and unique potential Identifiers"""
        top_graph = TopologyGraph.from_off_topology(topology)
        type_map = forcefield.run_atomtyping(top_graph, use_residue_map=False)
        for key, val in type_map.items():
            top_key = TopologyKey(atom_indices=(key,))
            self.slot_map[top_key] = PotentialKey(id=val["atomtype"])

    def store_potentials(self, forcefield: Forcefield) -> None:
        for top_key in self.slot_map:
            atom_params = forcefield.get_parameters(
                self.name, key=self.slot_map[top_key].id
            )

            atom_params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": u.kJ / u.mol, "sigma": u.nm},
            )

            self.potentials[self.slot_map[top_key]] = Potential(parameters=atom_params)


class FoyerConnectedAtomsHandler(PotentialHandler):
    connection_attribute: str = ""
    raise_on_missing_params = True

    def store_matches(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: Topology,
    ) -> None:
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

    def store_potentials(self, forcefield: Forcefield) -> None:
        for _, pot_key in self.slot_map.items():
            try:
                params = forcefield.get_parameters(
                    self.name, key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
                )
                params = self.get_params_with_units(params)
                self.potentials[pot_key] = Potential(parameters=params)
            except MissingForceError:
                # Here, we can safely assume that the ForceGenerator is Missing
                pass
            except MissingParametersError as e:
                if self.raise_on_missing_params:
                    raise e
                else:
                    pass

    @abstractmethod
    def get_params_with_units(self, params):
        raise NotImplementedError


class FoyerHarmonicBondHandler(FoyerConnectedAtomsHandler):
    name: str = "harmonic_bonds"
    expression: str = "1/2 * k * (r - length) ** 2"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    connection_attribute = "topology_bonds"

    def get_params_with_units(self, params):
        return _copy_params(
            params, param_units={"k": u.kJ / u.mol / u.nm ** 2, "length": u.nm}
        )


class FoyerHarmonicAngleHandler(FoyerConnectedAtomsHandler):
    name: str = "harmonic_angles"
    expression: str = "0.5 * k * (theta-angle)**2"
    independent_variables: Set[str] = {"theta"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    connection_attribute: str = "angles"

    def get_params_with_units(self, params):
        return _copy_params(
            {"k": params["k"], "angle": params["theta"]},
            param_units={
                "k": u.kJ / u.mol / u.nm ** 2,
                "angle": u.dimensionless,
            },
        )


class FoyerRBProperHandler(FoyerConnectedAtomsHandler):
    name: str = "rb_propers"
    expression: str = (
        "C0 * cos(phi)**0 + C1 * cos(phi)**1 + "
        "C2 * cos(phi)**2 + C3 * cos(phi)**3 + "
        "C4 * cos(phi)**4 + C5 * cos(phi)**5"
    )
    independent_variables: Set[str] = {"phi"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    connection_attribute: str = "propers"
    raise_on_missing_params: bool = False

    def get_params_with_units(self, params):
        rb_params = {k.upper(): v for k, v in params.items()}
        param_units = {k: u.kJ / u.mol for k in rb_params}
        return _copy_params(rb_params, param_units=param_units)


class FoyerRBImproperHandler(FoyerRBProperHandler):
    name: str = "rb_impropers"
    connection_attribute: str = "impropers"


class FoyerPeriodicProperHandler(FoyerConnectedAtomsHandler):
    name: str = "periodic_propers"
    expression: str = "k * (1 + cos(periodicity * phi - phase))"
    independent_variables: Set[str] = {"phi"}
    connection_attribute: str = "propers"
    raise_on_missing_params: bool = False

    def get_params_with_units(self, params):
        return _copy_params(
            params,
            param_units={
                "k": u.kJ / u.mol / u.nm ** 2,
                "phase": u.dimensionless,
                "periodicity": u.dimensionless,
            },
        )


class FoyerPeriodicImproperHandler(FoyerPeriodicProperHandler):
    name: str = "periodic_impropers"
    connection_attribute: str = "impropers"
