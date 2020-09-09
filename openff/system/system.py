from collections import OrderedDict
from typing import Dict, Optional, Union

import numpy as np
import pint
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField, ParameterHandler
from pydantic import BaseModel, root_validator, validator
from simtk.openmm.app import Topology as OpenMMTopology
from simtk.unit import Quantity as SimTKQuantity

from . import unit
from .exceptions import InvalidBoxError, ToolkitTopologyConformersNotFoundError
from .interop import parmed
from .types import UnitArray
from .typing.smirnoff.data import (
    SMIRNOFFTermCollection,
    build_slot_smirks_map,
    build_slot_smirks_map_term,
    build_smirks_potential_map_term,
)
from .utils import simtk_to_pint


def potential_map_from_terms(collection):
    """Builds a high-level smirks map from corresponding maps for each handler"""
    mapping = dict()

    for key, val in collection.terms.items():
        mapping[key] = val.smirks_map

    return mapping


class ProtoSystem(BaseModel):
    """
    A primitive object for other System objects to be built off of

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    topology : openforcefield.topology.Topology or simtk.openmm.app.topology.Topology
        A representation of the chemical topology of the system
    positions : UnitArray
        Positions of all atoms in the system
    box : UnitArray, optional
        Periodic box vectors. A value of None implies a non-periodic system
    """

    topology: Union[Topology, OpenMMTopology]
    positions: UnitArray
    box: Optional[UnitArray]

    # TODO: I needed to set pre=True to get this to override the Array type. This is bad
    # and instead this attribute should be handled by a custom class that deals with
    # all of the complexity (NumPy/simtk.unit.Quantity/pint.Quantity) and spits out
    # a single thing that plays nicely with things
    @validator("positions", "box", pre=True)
    def validate_in_space(cls, val):
        # TODO: More gracefully deal with None values
        if val is None:
            return val
        if isinstance(val, SimTKQuantity):
            val = UnitArray(simtk_to_pint(val))
        if isinstance(val, np.ndarray):
            return UnitArray(val, units=unit.nm)
        if isinstance(val, (pint.Quantity, UnitArray)):
            if val.dimensionless:
                units = unit.nm
            else:
                units = val.units
            if not units.is_compatible_with("nm"):
                raise pint.DimensionalityError(
                    units1="nm",
                    units2=units,
                    extra_msg=".\tBox vectors must be in length units.",
                )
            return UnitArray(val.magnitude, units=units)
        else:
            raise TypeError

    @validator("box")
    def validate_box(cls, val):
        if val is None:
            return val
        if val.shape == (3, 3):
            return val
        elif val.shape == (3,):
            return val * np.eye(3)
        else:
            raise InvalidBoxError

    @validator("topology", pre=True)
    def validate_topology(cls, val):
        if isinstance(val, Topology):
            return val
        elif isinstance(val, OpenMMTopology):
            return Topology.from_openmm(val)
        else:
            raise TypeError

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class System(ProtoSystem):
    """The OpenFF System object.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    topology : openforcefield.topology.Topology or simtk.openmm.app.topology.Topology
        A representation of the chemical topology of the system
    positions : UnitArray
        Positions of all atoms in the system
    box : UnitArray, optional
        Periodic box vectors. A value of None implies a non-periodic system
    forcefield : openforcefield.typing.engines.smirnoff.forcefield.ForceField or
        openforcefield.typing.engines.smirnoff.parameters.ParameterHandler, optional
        A SMIRNOFF force field or portion thereof as a parameter handler
    slot_smirks_map : dict, optional
        A nested dictionary mapping, for each handler, slot identifiers to SMIRKS
        patterns
    smirks_potential_map : dict, optional
        A nested dictionary mapping, for each handler, SMIRKS patterns to instantiated
        potential objects
    term_colection : openff.system.typing.smirnoff.data.SMIRNOFFTermCollection, optional
        A collection of instantiated potential terms from a SMIRNOFF force field
    """

    forcefield: Union[ForceField, ParameterHandler] = None
    slot_smirks_map: Dict = dict()
    smirks_potential_map: Dict = dict()
    term_collection: SMIRNOFFTermCollection = None

    @classmethod
    def from_proto_system(
        cls,
        proto_system,
        forcefield=None,
        slot_smirks_map=dict(),
        smirks_potential_map=dict(),
        term_collection=SMIRNOFFTermCollection(),
    ):
        """Construct a System from an existing ProtoSystem and other parameters"""
        return cls(
            topology=proto_system.topology,
            positions=proto_system.positions,
            box=proto_system.box,
            forcefield=forcefield,
            slot_smirks_map=slot_smirks_map,
            smirks_potential_map=smirks_potential_map,
            term_collection=term_collection,
        )

    @classmethod
    def from_toolkit(cls, topology, forcefield):
        """Attempt to construct a System from a toolkit Topology and ForceField"""
        positions = None
        for top_mol in topology.topology_molecules:
            if not top_mol.reference_molecule.conformers:
                raise ToolkitTopologyConformersNotFoundError(top_mol)
            mol_pos = simtk_to_pint(top_mol.reference_molecule.conformers[0])
            if positions is None:
                positions = mol_pos
            else:
                positions = np.vstack([positions, mol_pos])

        return cls(
            topology=topology,
            forcefield=forcefield,
            positions=positions,
            box=simtk_to_pint(topology.box_vectors),
        )

    @root_validator
    def validate_forcefield_data(cls, values):
        # TODO: Replace this messy logic with something cleaner
        if not values["forcefield"]:
            if not values["slot_smirks_map"] or not values["smirks_potential_map"]:
                pass  # raise TypeError('not given an ff, need maps')
        if values["forcefield"]:
            # TODO: Let other typing engines drop in here
            values["slot_smirks_map"] = build_slot_smirks_map(
                forcefield=values["forcefield"], topology=values["topology"]
            )
            values["term_collection"] = SMIRNOFFTermCollection.from_toolkit_data(
                toolkit_forcefield=values["forcefield"],
                toolkit_topology=values["topology"],
            )
            values["smirks_potential_map"] = potential_map_from_terms(
                values["term_collection"]
            )
        return values

    # TODO: These valiators pretty much don't do anything now
    @validator("forcefield")
    def validate_forcefield(cls, val):
        if not val:
            return val
        if isinstance(val, ForceField):
            return val
        if isinstance(val, OrderedDict):
            # TODO: Make this the default drop-in if the toolkit reworks ForceField to be more dict-like
            forcefield = ForceField()
            forcefield._load_smirnoff_data(val)
            return forcefield
        if isinstance(val, str):
            return ForceField(val)
        else:
            raise TypeError

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def apply_single_handler(self, handler):
        """Apply a single parameter handler to a system."""
        # TODO: Abstract this away to be SMIRNOFF-agnostic
        if handler._TAGNAME == "Electrostatics":
            raise NotImplementedError()

        self.slot_smirks_map[handler._TAGNAME] = build_slot_smirks_map_term(
            handler=handler,
            topology=self.topology,
        )

        self.smirks_potential_map[handler._TAGNAME] = build_smirks_potential_map_term(
            handler=handler,
            topology=self.topology,
            forcefield=self.forcefield,
        )

        self.term_collection.add_parameter_handler(
            handler, topology=self.topology, forcefield=None
        )

    def to_file(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_parmed(self):
        return parmed.to_parmed(self)

    def to_openmm(self):
        raise NotImplementedError()
