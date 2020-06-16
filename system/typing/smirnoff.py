from warnings import warn
from typing import Dict

from pydantic import BaseModel
import numpy as np

from ..collections import PotentialHandler, PotentialCollection
from ..utils import simtk_to_pint
from ..potential import ParametrizedAnalyticalPotential as Potential
from ..exceptions import SMIRNOFFHandlerNotImplementedError

# TODO: Probably shouldn't have this as a global variable floating around
SUPPORTED_HANDLERS = {'vdW', 'Bonds', 'Angles'}


def build_slot_smirks_map(topology, forcefield):
    """Turn a SMIRNOFF force field into a mapping between slots and SMIRKS"""
    typing_map = dict()

    for handler in forcefield._parameter_handlers.keys():
        typing_map[handler] = build_slot_smirks_map_term(handler, forcefield, topology)

    return typing_map


def build_slot_smirks_map_term(handler, forcefield, topology):
    """Get mapping between slot keys and SMIRKS"""
    slot_potential_map = dict()

    matches = forcefield.get_parameter_handler(handler).find_matches(topology)

    for atom_key, atom_match in matches.items():
        slot_potential_map[atom_key] = atom_match.parameter_type.smirks

    return slot_potential_map


def build_smirks_potential_map(name, forcefield, smirks_map=None):
    """Temporary stand-in for .to_potential calls in toolkit ParameterHandler objects."""
    if name not in SUPPORTED_HANDLERS:
        warn(f'handler {name} not implemented')
        return potential_collection

    if name == 'vdW':
        return build_smirks_potential_map_vdw(forcefield=forcefield, smirks_map=smirks_map)
    if name == 'Bonds':
        return build_smirks_potential_map_bonds(forcefield=forcefield, smirks_map=smirks_map)
    if name == 'Angles':
        return build_smirks_potential_map_angles(forcefield=forcefield, smirks_map=smirks_map)


def build_smirks_potential_map_vdw(forcefield, smirks_map=None):
    # potential_collection.handlers.update({'vdW': PotentialHandler(name='vdW')})
    mapping = dict()

    for param in forcefield.get_parameter_handler('vdW').parameters:
        if smirks_map is not None:
            if param.smirks not in smirks_map.values():
                continue
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

        mapping[param.smirks] = potential

    return mapping


def build_smirks_potential_map_bonds(forcefield, smirks_map=None):
    mapping = dict()

    for param in forcefield.get_parameter_handler('Bonds').parameters:
        if smirks_map is not None:
            if param.smirks not in smirks_map.values():
                continue
        k = simtk_to_pint(param.k)
        length = simtk_to_pint(param.length)

        potential = Potential(
            name=param.id,
            smirks=param.smirks,
            expression='1/2*k*(length-length_0)**2',
            independent_variables={'length_0'},
            parameters={'k': k, 'length': length},
        )

        mapping[param.smirks] = potential

    return mapping

def build_smirks_potential_map_angles(forcefield, smirks_map=None):
    mapping = dict()

    for param in forcefield.get_parameter_handler('Angles').parameters:
        if smirks_map is not None:
            if param.smirks not in smirks_map.values():
                continue
        k = simtk_to_pint(param.k)
        angle = simtk_to_pint(param.angle)

        potential = Potential(
            name=param.id,
            smirks=param.smirks,
            expression='1/2*k*(angle-angle_0)**2',
            independent_variables={'angle_0'},
            parameters={'k': k, 'angle': angle},
        )

        mapping[param.smirks] = potential

    return mapping


def build_smirnoff_collection(forcefield, smirks_map=None):
    """Build a PotentialCollection storing data in a SMIRNOFF force field."""
    handlers = [h for h in forcefield._parameter_handlers.keys() if h in SUPPORTED_HANDLERS]

    smirnoff_collection = PotentialCollection()

    for handler in handlers:
        smirnoff_collection = add_handler(forcefield, smirnoff_collection, handler, smirks_map=smirks_map)

    return smirnoff_collection


class SMIRNOFFPotentialTerm(BaseModel):
    """
    Base class for handling terms in a potential energy function,
    adhering to the functionality of the SMIRNOFF specification.
    """

    name: str
    smirks_map: Dict[tuple, str] = dict()
    potentials: Dict[str, Potential] = dict()

    @classmethod
    def build_from_toolkit_data(cls, name, forcefield, topology):
        term = cls(name=name)
        term.smirks_map = build_slot_smirks_map_term(name, forcefield=forcefield, topology=topology)
        term.potentials = build_smirks_potential_map(name, forcefield=forcefield, smirks_map=term.smirks_map)
        return term

    def smirks_map_to_atom_indices(self):
        return np.array([val[0] for val in self.smirks_map[self.name].keys()])

    def term_to_flattened_array(self):
        raise NotImplementedError

    def parametrize(self):
        return smirks_map_to_flattened_array(p=p, smirks_map=smirks_map, mapping=mapping)

    class Config:
        arbitrary_types_allowed = True


class SMIRNOFFvdWTerm(SMIRNOFFPotentialTerm):

    name: str = 'vdW'

#    @classmethod
#    def build_from_toolkit_data(cls, name, forcefield, topology):
#        term = cls(name=name)
#        term.smirks_map = build_slot_smirks_map_term(name, forcefield=forcefield, topology=topology)
#        term.potentials = build_smirks_potential_map(name, forcefield=forcefield, smirks_map=self.smirks_map)

    def get_p(self):
        """get p from a SMIRNOFFPotentialTerm
        returns

            p : flattened representation of force field parameters (for this term in a potential energy expression).
                indices in this array are freshly created
            mapping : dict mapping smirks patterns (for this term) to their 'id' that was just generated
        """
        p = []
        mapping = dict()
        for i, pot in enumerate(self.potentials.values()):
            p.append([pot.parameters['sigma'].magnitude, pot.parameters['epsilon'].magnitude])
            mapping.update({pot.smirks: i})
        return np.array(p), mapping


potential_term_mapping = {'vdW': SMIRNOFFvdWTerm}

class SMIRNOFFTermCollection(BaseModel):

    terms: Dict[str, SMIRNOFFPotentialTerm] = dict()

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_toolkit_data(cls, toolkit_forcefield, toolkit_topology):
        collection = cls()
        for handler in toolkit_forcefield._parameter_handlers.keys():
            if handler not in SUPPORTED_HANDLERS:
                raise SMIRNOFFHandlerNotImplementedError(handler)
            if handler not in potential_term_mapping.keys():
                raise SMIRNOFFHandlerNotImplementedError(handler)
            if handler in potential_term_mapping.keys():
                term = potential_term_mapping[handler]()
                collection.terms[handler] = term.build_from_toolkit_data(
                    name=handler,
                    forcefield=toolkit_forcefield,
                    topology=toolkit_topology
                )
        return collection
