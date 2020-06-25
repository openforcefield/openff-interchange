from warnings import warn
from typing import Dict
from functools import partial

from pydantic import BaseModel
import numpy as np

from .. import unit
from ..utils import simtk_to_pint, jax_available, get_partial_charges_from_openmm_system
from ..potential import ParametrizedAnalyticalPotential as Potential
from ..exceptions import SMIRNOFFHandlerNotImplementedError, JAXNotInstalledError

# TODO: Probably shouldn't have this as a global variable floating around
SUPPORTED_HANDLERS = {'vdW', 'Bonds', 'Angles', 'Electrostatics'}


def build_slot_smirks_map(topology, forcefield):
    """Turn a SMIRNOFF force field into a mapping between slots and SMIRKS"""
    typing_map = dict()

    for handler in forcefield._parameter_handlers.keys():
        typing_map[handler] = build_slot_smirks_map_term(handler, forcefield, topology)

    return typing_map


def build_slot_smirks_map_term(handler, forcefield, topology):
    """Get mapping between slot keys and SMIRKS for only one term"""
    slot_smirks_map = dict()

    if handler == 'Electrostatics':
        return dummy_atomic_slots_map(topology)

    matches = forcefield.get_parameter_handler(handler).find_matches(topology)

    for atom_key, atom_match in matches.items():
        slot_smirks_map[atom_key] = atom_match.parameter_type.smirks

    return slot_smirks_map


def dummy_atomic_slots_map(topology):
    """
    Return something that looks like a slot -> SMIRKS map, but the SMIRKS patterns
    are actually just atom indices as strings
    """
    mapping = dict()

    for idx, atom in enumerate(topology.topology_atoms):
        mapping[(idx,)] = str(idx)
    return mapping


def build_smirks_potential_map(forcefield, smirks_map=None):
    mapping = dict()

    for handler in forcefield._parameter_handlers.keys():
        if handler not in SUPPORTED_HANDLERS:
            continue
        if smirks_map:
            partial_smirks_map = smirks_map[handler]
        else:
            partial_smirks_map = None
        mapping[handler] = build_smirks_potential_map_term(handler, forcefield, partial_smirks_map)

    return mapping


def build_smirks_potential_map_term(name, forcefield, smirks_map=None, topology=None):
    """Temporary stand-in for .to_potential calls in toolkit ParameterHandler objects."""
    if name not in SUPPORTED_HANDLERS:
        warn(f'handler {name} not implemented')
        raise Exception # return potential_collection

    if name == 'vdW':
        return build_smirks_potential_map_vdw(forcefield=forcefield, smirks_map=smirks_map)
    if name == 'Bonds':
        return build_smirks_potential_map_bonds(forcefield=forcefield, smirks_map=smirks_map)
    if name == 'Angles':
        return build_smirks_potential_map_angles(forcefield=forcefield, smirks_map=smirks_map)
    if name == 'Electrostatics':
        return build_smirks_potential_map_electrostatics(forcefield=forcefield, topology=topology, smirks_map=smirks_map)


def build_smirks_potential_map_vdw(forcefield, smirks_map=None):
    mapping = dict()

    for param in forcefield.get_parameter_handler('vdW').parameters:
        if not smirks_map:
            if param.smirks not in smirks_map.values():
                continue
        if not param.sigma:
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
        if smirks_map:
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
        if not smirks_map:
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


def build_smirks_potential_map_electrostatics(forcefield, topology, smirks_map=None):
    """
    Note: This mapping does not go through SMIRKS and should be replaced with future toolkit features;
    See https://github.com/openforcefield/openforcefield/issues/619

    Note: This mapping does not store an interaction term, it only stores the partial charge
    """
    mapping = dict()

    if not smirks_map:
        smirks_map = build_slot_smirks_map_term('Electrostatics', forcefield=forcefield, topology=topology)

    partial_charges = get_partial_charges_from_openmm_system(forcefield.create_openmm_system(topology))

    for key, val in smirks_map.items():
        mapping[val] = partial_charges[int(val)]

    return mapping


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
        term.potentials = build_smirks_potential_map_term(name, forcefield=forcefield, smirks_map=term.smirks_map)
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

    @classmethod
    def build_from_toolkit_data(cls, name, forcefield, topology):
        term = cls(name=name)
        term.smirks_map = build_slot_smirks_map_term(name, forcefield=forcefield, topology=topology)
        term.potentials = build_smirks_potential_map_term(name, forcefield=forcefield, smirks_map=term.smirks_map)
        return term

    def get_p(self, use_jax=False):
        """get p from a SMIRNOFFPotentialTerm
        returns

            p : flattened representation of force field parameters (for this term in a potential energy expression).
                indices in this array are freshly created
            mapping : dict mapping smirks patterns (for this term) to their 'id' that was just generated
        """
        if use_jax:
            import jax.numpy as np
        else:
            import numpy as np

        p = []
        mapping = dict()
        for i, pot in enumerate(self.potentials.values()):
            p.append([pot.parameters['sigma'].magnitude, pot.parameters['epsilon'].magnitude])
            mapping.update({pot.smirks: i})
        return np.array(p), mapping

    def get_q(self, p=None, mapping=None, use_jax=False):
        if use_jax:
            import jax.numpy as np
        else:
            import numpy as np

        if None in (p, mapping):
            (p, mapping) = self.get_p()
        q = []
        for i, val in enumerate(self.smirks_map.keys()):
            q.append(p[mapping[self.smirks_map[(i,)]]])
        return np.array(q)

    def parametrize(self, p=None, smirks_map=None, mapping=None):
        if None in (p, mapping):
            (p, mapping) = self.get_p(use_jax=True)
        return self.get_q(p=p, mapping=mapping, use_jax=True)

    def parametrize_partial(self):
        return partial(self.parametrize, smirks_map=self.smirks_map, mapping=self.get_p()[1])

    # TODO: Don't use JAX so forcefully when calling other functions from here
    def get_param_matrix(self):
        if not jax_available:
            raise JAXNotInstalledError
        import jax
        (p, mapping) = self.get_p(use_jax=True)
        parametrize_partial = partial(self.parametrize, smirks_map=self.smirks_map, mapping=mapping)

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])

class SMIRNOFFBondTerm(SMIRNOFFPotentialTerm):

    name: str = 'Bonds'


class SMIRNOFFAngleTerm(SMIRNOFFPotentialTerm):

    name: str = 'Angles'


class ElectrostaticsTerm(SMIRNOFFPotentialTerm):

    name: str = 'Electrostatics'
    smirks_map: Dict[tuple, str] = dict()
    potentials: Dict[str, unit.Quantity] = dict()

    @classmethod
    def build_from_toolkit_data(cls, name, forcefield, topology):

        term = cls(name=name)
        term.smirks_map = build_slot_smirks_map_term(name, forcefield=forcefield, topology=topology)
        term.potentials = build_smirks_potential_map_electrostatics(forcefield=forcefield, topology=topology, smirks_map=term.smirks_map)
        return term

potential_term_mapping = {
    'vdW': SMIRNOFFvdWTerm,
    'Bonds': SMIRNOFFBondTerm,
    'Angles': SMIRNOFFAngleTerm,
    'Electrostatics': ElectrostaticsTerm,
}


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
