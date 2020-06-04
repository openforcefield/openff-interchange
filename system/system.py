from typing import Union, Iterable, List, Dict

from pydantic import BaseModel, validator
import pint

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Topology as ToolkitTopology
from openforcefield.topology.molecule import Atom as ToolkitAtom

from .potential import ParametrizedAnalyticalPotential as Potential
from .typing.smirnoff import build_slots_parameter_map
from .utils import simtk_to_pint


u = pint.UnitRegistry()


class PotentialHandler(BaseModel):

    name: str
    potentials: Dict[str, Potential] = None

    def __getitem__(self, potential_smirks):
        return self.potentials[potential_smirks]

def handler_conversion(forcefield, potential_collection, handler_name):
    """Temporary stand-in for .to_potential calls in toolkit ParameterHandler objects."""
    if handler_name != 'vdW':
        raise NotImplementedError

    for param in forcefield.get_parameter_handler(handler_name).parameters:
        if param.sigma is None:
            sigma = 2. * param.rmin_half / (2.**(1. / 6.))
        else:
            sigma = param.sigma
        sigma = simtk_to_pint(sigma)
        epsilon = simtk_to_pint(param.epsilon)

        potential = Potential(
            name=param.id,
            smirks=param.smirks,
            expression='4*epsilon*((sigma/r)**12-(sigma/r)**6)',
            independent_variables={'r'},
            parameters={'sigma': sigma, 'epsilon': epsilon},
        )

        try:
            potential_collection.handlers['vdW'][param.smirks] = potential
        except (AttributeError, TypeError):
            potential_collection.handlers = {
                'vdW': PotentialHandler(
                    name='vdW',
                    potentials={
                        param.smirks: potential,
                    }
                )

            }

    return potential_collection


class PotentialCollection(BaseModel):

    handlers: Dict[str, PotentialHandler] = None

    @classmethod
    def from_toolkit_forcefield(cls, toolkit_forcefield):

        toolkit_handlers = toolkit_forcefield._parameter_handlers.keys()
        supported_handlers = ['vdW']

        for handler in toolkit_handlers:
            if handler not in supported_handlers:
                continue
            handler_conversion(toolkit_forcefield, cls, handler)
        return cls

    def __getitem__(self, handler_name):
        return self.handlers[handler_name]


class Atom(ToolkitAtom):

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, atomic_number, parameter_id=None):
        super().__init__(
            atomic_number=atomic_number,
            formal_charge=0,
            is_aromatic=False,
        )
        self._parameter_id = parameter_id

    @property
    def atomic_number(self):
        return self._atomic_number

    @property
    def parameter_id(self):
        return self._parameter_id

    @parameter_id.setter
    def parameter_id(self, potential):
        self._parameter_id = potential


class Topology(BaseModel):

    atoms: List[ToolkitAtom]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_toolkit_topology(cls, toolkit_topology):

        atoms = []

        for atom in toolkit_topology.topology_atoms:
            atoms.append(atom.atom)

        return cls(atoms=atoms)


class System(BaseModel):
    """The OpenFF System object."""

    topology: Union[Topology, ToolkitTopology]
    potential_collection: Union[PotentialCollection, ForceField]
    positions: Iterable = None
    box: Iterable = None
    slots_map: Dict = None

    @validator("potential_collection")
    def validate_forcefield(cls, val):
        if isinstance(val, ForceField):
            return PotentialCollection.from_toolkit_forcefield(val)
        elif isinstance(val, PotentialCollection):
            return val
        else:
            raise TypeError

    @validator("topology")
    def validate_topology(cls, val):
        if isinstance(val, ToolkitTopology):
            return val
        elif isinstance(val, Topology):
            return NotImplementedError
        else:
            raise TypeError

    @validator("*")
    def dummy_validator(cls, val):
        return val

    class Config:
        arbitrary_types_allowed = True

    def run_typing(self, toolkit_forcefield, toolkit_topology):
        """Just store the slots map"""
        self.slots_map = build_slots_parameter_map(
            forcefield=toolkit_forcefield,
            topology=toolkit_topology
        )

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        raise NotImplementedError()

    def to_openmm(self):
        raise NotImplementedError()
