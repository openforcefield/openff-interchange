from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Dict, Set, Type

from ele import element_from_atomic_number
from openff.units import unit
from openff.utilities.utils import has_pkg, requires_package

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.system import System
from openff.system.models import PotentialKey, TopologyKey
from openff.system.types import FloatQuantity

if TYPE_CHECKING:
    from foyer.forcefield import Forcefield
    from foyer.topology_graph import TopologyGraph

    from openff.system.components.misc import OFFBioTop

# Is this the safest way to achieve PotentialKey id separation?
POTENTIAL_KEY_SEPARATOR = "-"


if has_pkg("foyer"):
    from foyer.topology_graph import TopologyGraph  # noqa

    @classmethod  # type: ignore
    def from_off_topology(cls, off_topology: "OFFBioTop") -> "TopologyGraph":
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


@requires_package("foyer")
def from_foyer(topology: "OFFBioTop", ff: "Forcefield", **kwargs) -> System:
    system = System()
    system.topology = topology

    for name, Handler in get_handlers_callable().items():
        system.handlers[name] = Handler()

    system.handlers["vdW"].store_matches(ff, topology=topology)
    system.handlers["vdW"].store_potentials(forcefield=ff)  # type: ignore

    atom_slots = system.handlers["vdW"].slot_map

    system.handlers["Electrostatics"].store_charges(  # type: ignore[attr-defined]
        atom_slots=atom_slots,
        forcefield=ff,
    )

    # TODO: Replace with API points after https://github.com/mosdef-hub/foyer/issues/397:
    from simtk.openmm.app.forcefield import NonbondedGenerator  # type: ignore

    nonbonded_generator = ff.get_generator(ff, gen_type=NonbondedGenerator)
    system.handlers["vdW"].scale_14 = nonbonded_generator.lj14scale  # type: ignore[attr-defined]
    system.handlers["Electrostatics"].scale_14 = nonbonded_generator.coulomb14scale  # type: ignore[attr-defined]

    for name, handler in system.handlers.items():
        if name not in ["vdW", "Electrostatics"]:
            handler.store_matches(atom_slots, topology=topology)
            handler.store_potentials(ff)

    return system


def get_handlers_callable() -> Dict[str, Type[PotentialHandler]]:
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
    name: str = "atoms"
    expression: str = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"
    independent_variables: Set[str] = {"r"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5  # TODO: Replace with Foyer API point?
    scale_15: float = 1.0
    method: str = "cutoff"
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom  # type: ignore

    def store_matches(
        self,
        forcefield: "Forcefield",
        topology: "OFFBioTop",
    ) -> None:
        """Populate slotmap with key-val pairs of slots and unique potential Identifiers"""
        from foyer.atomtyper import find_atomtypes

        top_graph = TopologyGraph.from_off_topology(topology)
        type_map = find_atomtypes(top_graph, forcefield=forcefield)
        for key, val in type_map.items():
            top_key = TopologyKey(atom_indices=(key,))
            self.slot_map[top_key] = PotentialKey(id=val["atomtype"])

    def store_potentials(self, forcefield: "Forcefield") -> None:
        for top_key in self.slot_map:
            atom_params = forcefield.get_parameters(
                self.name, key=self.slot_map[top_key].id
            )

            atom_params = _copy_params(
                atom_params,
                "charge",
                param_units={"epsilon": unit.kJ / unit.mol, "sigma": unit.nm},
            )

            self.potentials[self.slot_map[top_key]] = Potential(parameters=atom_params)


class FoyerElectrostaticsHandler(PotentialHandler):
    name: str = "Electrostatics"
    method: str = "PME"
    expression: str = "coul"
    independent_variables: Set[str] = {"r"}
    charges: Dict[TopologyKey, float] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5  # TODO: Replace with Foyer API point?
    scale_15: float = 1.0
    cutoff: FloatQuantity["angstrom"] = 9.0 * unit.angstrom  # type: ignore

    def store_charges(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        forcefield: "Forcefield",
    ):
        for top_key, pot_key in atom_slots.items():
            foyer_params = forcefield.get_parameters("atoms", pot_key.id)
            charge = foyer_params["charge"]
            charge = charge * unit.elementary_charge
            self.charges[top_key] = charge


class FoyerConnectedAtomsHandler(PotentialHandler):
    connection_attribute: str = ""
    raise_on_missing_params = True

    def store_matches(
        self,
        atom_slots: Dict[TopologyKey, PotentialKey],
        topology: "OFFBioTop",
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

    def store_potentials(self, forcefield: "Forcefield") -> None:
        from foyer.exceptions import MissingForceError, MissingParametersError

        for _, pot_key in self.slot_map.items():
            try:
                params = forcefield.get_parameters(
                    self.name, key=pot_key.id.split(POTENTIAL_KEY_SEPARATOR)
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
            params,
            param_units={"k": unit.kJ / unit.mol / unit.nm ** 2, "length": unit.nm},
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
                "k": unit.kJ / unit.mol / unit.radian ** 2,
                "angle": unit.dimensionless,
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
        param_units = {k: unit.kJ / unit.mol for k in rb_params}
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
                "k": unit.kJ / unit.mol / unit.nm ** 2,
                "phase": unit.dimensionless,
                "periodicity": unit.dimensionless,
            },
        )


class FoyerPeriodicImproperHandler(FoyerPeriodicProperHandler):
    name: str = "periodic_impropers"
    connection_attribute: str = "impropers"
