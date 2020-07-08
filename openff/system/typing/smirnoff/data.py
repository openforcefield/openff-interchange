from warnings import warn
from typing import Dict
from functools import partial

from pydantic import BaseModel
import numpy as np

from openforcefield.typing.engines.smirnoff import ForceField

from ... import unit
from ...utils import simtk_to_pint, jax_available, get_partial_charges_from_openmm_system
from ...potential import ParametrizedAnalyticalPotential as Potential
from ...exceptions import SMIRNOFFHandlerNotImplementedError, JAXNotInstalledError


def build_slot_smirks_map(topology, forcefield):
    """Generate a mapping between slots and SMIRKS patterns given an OpenFF ForceField"""
    typing_map = dict()

    for handler_name, handler in forcefield._parameter_handlers.items():
        typing_map[handler_name] = build_slot_smirks_map_term(handler, topology)

    return typing_map


def build_slot_smirks_map_term(handler, topology):
    """Generate a mapping between slots and SMIRKS patterns for one parameter handler in an OpenFF ForceField"""
    slot_smirks_map = dict()

    if handler._TAGNAME == 'Electrostatics':
        return dummy_atomic_slots_map(topology)

    matches = handler.find_matches(topology)

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
    """Generate a mapping between SMIRKS patterns and potential objects for an entire OpenFF ForceField"""
    mapping = dict()

    for handler in forcefield._parameter_handlers.keys():
        if handler not in SUPPORTED_HANDLER_MAPPING.keys():
            continue
        if smirks_map:
            partial_smirks_map = smirks_map[handler]
        else:
            partial_smirks_map = None
        mapping[handler] = build_smirks_potential_map_term(forcefield[handler], partial_smirks_map)

    return mapping


def build_smirks_potential_map_term(handler, smirks_map=None, topology=None, forcefield=None):
    """Generate a mapping between SMIRKS patterns and potential objects for a single parameter handler in an OpenFF ForceField"""
    if handler._TAGNAME not in SUPPORTED_HANDLER_MAPPING.keys():
        warn(f'handler {name} not implemented')
        raise Exception # return potential_collection

    if handler._TAGNAME == 'vdW':
        return build_smirks_potential_map_vdw(handler=handler, smirks_map=smirks_map)
    if handler._TAGNAME == 'Bonds':
        return build_smirks_potential_map_bonds(handler=handler, smirks_map=smirks_map)
    if handler._TAGNAME == 'Angles':
        return build_smirks_potential_map_angles(handler=handler, smirks_map=smirks_map)
    if handler._TAGNAME == 'ProperTorsions':
        return build_smirks_potential_map_propers(handler=handler, smirks_map=smirks_map)
    if handler._TAGNAME == 'ImproperTorsions':
        return build_smirks_potential_map_impropers(handler=handler, smirks_map=smirks_map)
    if handler._TAGNAME == 'Electrostatics':
        return build_smirks_potential_map_electrostatics(forcefield=forcefield, topology=topology, smirks_map=smirks_map)


def build_smirks_potential_map_vdw(handler, smirks_map=None):
    mapping = dict()

    for param in handler.parameters:
        if smirks_map:
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


def build_smirks_potential_map_bonds(handler, smirks_map=None):
    mapping = dict()

    for param in handler.parameters:
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

def build_smirks_potential_map_angles(handler, smirks_map=None):
    mapping = dict()

    for param in handler.parameters:
        if smirks_map:
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

def build_smirks_potential_map_propers(handler, smirks_map=None):
    # TODO: Properly deal with arbitrary values of n
    mapping = dict()

    def expr_from_n(n):
        return f'k{n}*(1+cos(periodicity{n}*theta-phase{n}))'

    for param in handler.parameters:
        if smirks_map:
            if param.smirks not in smirks_map.values():
                continue

        expr = '0'
        params = dict()

        for n in range(3):
            try:
                param.k[n]
            except IndexError:
                continue
            expr += '+' + expr_from_n(n+1)
            params.update({
                f'k{n+1}': simtk_to_pint(param.k[n]),
                f'periodicity{n+1}': param.periodicity[n] * unit.dimensionless,
                f'phase{n+1}': simtk_to_pint(param.phase[n]),
            })

        potential = Potential(
            name=param.id,
            smirks=param.smirks,
            expression=expr,
            independent_variables={'theta'},
            parameters=params,
        )

        mapping[param.smirks] = potential

    return mapping

def build_smirks_potential_map_impropers(handler, smirks_map=None):
    # TODO: Properly deal with arbitrary values of n
    mapping = dict()

    def expr_from_n(n):
        return f'k{n}*(1+cos(periodicity{n}*theta-phase{n}))'

    for param in handler.parameters:
        if smirks_map:
            if param.smirks not in smirks_map.values():
                continue

        expr = '0'
        params = {}

        for n in range(3):
            try:
                param.k[n]
            except IndexError:
                continue
            expr += '+' + expr_from_n(n+1)
            params.update({
                f'k{n+1}': simtk_to_pint(param.k[n]),
                f'periodicity{n+1}': param.periodicity[n] * unit.dimensionless,
                f'phase{n+1}': simtk_to_pint(param.phase[n]),
            })

        potential = Potential(
            name=param.id,
            smirks=param.smirks,
            expression=expr,
            independent_variables={'theta'},
            parameters=params,
        )

        mapping[param.smirks] = potential

    return mapping

def build_smirks_potential_map_electrostatics(forcefield, topology, smirks_map=None, partial_charges=None):
    """
    Build a mapping between SMIRKS patterns and partial charges

    Note: This mapping does not go through SMIRKS and should be replaced with future toolkit features;
    See https://github.com/openforcefield/openforcefield/issues/619

    Note: This mapping does not store an interaction term, it only stores the partial charge
    """
    mapping = dict()

    if not smirks_map:
        smirks_map = build_slot_smirks_map_term(forcefield['Electrostatics'], topology=topology)

    # TODO: get partial charges from just a (single) electrostatics handler
    # Note: Requires some toolkit changes, something like
    # partial_charges = get_partial_charges_from_openmm_system(handler.get_partial_charges(topology))
    if partial_charges is None:
        partial_charges = get_partial_charges_from_openmm_system(forcefield.create_openmm_system(topology))

    for key, val in smirks_map.items():
        mapping[val] = partial_charges[int(key[0])]

    return mapping


class SMIRNOFFPotentialTerm(BaseModel):
    """
    Base class for handling terms in a potential energy function according to the SMIRNOFF specification

    Parameters
    ----------
    name : str
        A string identifier, often the id tag of the parameter
    smirks_map : dict of [tuple : str]
        Mapping between the slots, as tuples, and corresponding SMIRKS patterns
    potentials : dict of [str : openff.system.potential.Potential]
        Mapping between SMIRKS patterns and corresponding potential objects
    """

    name: str
    smirks_map: Dict[tuple, str] = dict()
    potentials: Dict[str, Potential] = dict()

    @classmethod
    def build_from_toolkit_data(cls, handler, topology, forcefield=None):
        """Construct a SMIRNOFFPotentialTerm from OpenFF Toolkit objects"""
        term = cls(name=handler._TAGNAME)
        term.smirks_map = build_slot_smirks_map_term(handler=handler, topology=topology)
        term.potentials = build_smirks_potential_map_term(handler=handler, smirks_map=term.smirks_map)
        return term

    def smirks_map_to_atom_indices(self):
        return np.array([val[0] for val in self.smirks_map[self.name].keys()])

    def get_force_field_parameters(self):
        raise NotImplementedError

    def get_system_parameters(self):
        raise NotImplementedError

    def parametrize(self):
        raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class SMIRNOFFvdWTerm(SMIRNOFFPotentialTerm):

    name: str = 'vdW'
    scale12: float = 0.0
    scale13: float = 0.0
    scale14: float = 0.5

    @classmethod
    def build_from_toolkit_data(cls, handler, topology, forcefield=None):
        term = cls(name=handler._TAGNAME)
        term.scale12 = handler.scale12
        term.scale13 = handler.scale13
        term.scale14 = handler.scale14
        term.smirks_map = build_slot_smirks_map_term(handler=handler, topology=topology)
        term.potentials = build_smirks_potential_map_term(handler=handler, smirks_map=term.smirks_map)
        return term

    def get_force_field_parameters(self, use_jax=False):
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
            p.append(pot.parameters['sigma'].magnitude)
            p.append(pot.parameters['epsilon'].magnitude)
            mapping.update({pot.smirks: i})
        return np.array(p), mapping

    def get_system_parameters(self, p=None, mapping=None, use_jax=False):
        if use_jax:
            import jax.numpy as np
        else:
            import numpy as np

        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=use_jax)
        q = []
        for i, val in enumerate(self.smirks_map.keys()):
            q.append(p[mapping[self.smirks_map[(i,)]]])
        return np.array(q)

    def parametrize(self, p=None, smirks_map=None, mapping=None):
        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=True)
        return self.get_system_parameters(p=p, mapping=mapping, use_jax=True)

    def parametrize_partial(self):
        return partial(self.parametrize, smirks_map=self.smirks_map, mapping=self.get_force_field_parameters()[1])

    # TODO: Don't use JAX so forcefully when calling other functions from here
    def get_param_matrix(self):
        if not jax_available:
            raise JAXNotInstalledError
        import jax
        (p, mapping) = self.get_force_field_parameters(use_jax=True)
        parametrize_partial = partial(self.parametrize, smirks_map=self.smirks_map, mapping=mapping)

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])

class SMIRNOFFBondTerm(SMIRNOFFPotentialTerm):

    name: str = 'Bonds'

    def get_force_field_parameters(self, use_jax=False):
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
            p.append(pot.parameters['length'].magnitude)
            p.append(pot.parameters['k'].magnitude)
            mapping.update({pot.smirks: i})
        return np.array(p), mapping

    def get_system_parameters(self, p=None, mapping=None, use_jax=False):
        if use_jax:
            import jax.numpy as np
        else:
            import numpy as np

        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=use_jax)
        q = []
        for key, val in self.smirks_map.items():
            q.append(p[mapping[val]])
        return np.array(q)

    def parametrize(self, p=None, smirks_map=None, mapping=None):
        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=True)
        return self.get_system_parameters(p=p, mapping=mapping, use_jax=True)

    def parametrize_partial(self):
        return partial(self.parametrize, smirks_map=self.smirks_map, mapping=self.get_force_field_parameters()[1])

    def get_param_matrix(self):
        if not jax_available:
            raise JAXNotInstalledError
        import jax
        (p, mapping) = self.get_force_field_parameters(use_jax=True)
        parametrize_partial = partial(self.parametrize, smirks_map=self.smirks_map, mapping=mapping)

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])



class SMIRNOFFAngleTerm(SMIRNOFFPotentialTerm):

    name: str = 'Angles'

    def get_force_field_parameters(self, use_jax=False):
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
            p.append(pot.parameters['angle'].magnitude)
            p.append(pot.parameters['k'].magnitude)
            mapping.update({pot.smirks: i})
        return np.array(p), mapping

    def get_system_parameters(self, p=None, mapping=None, use_jax=False):
        if use_jax:
            import jax.numpy as np
        else:
            import numpy as np

        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=use_jax)
        q = []
        for key, val in self.smirks_map.items():
            q.append(p[mapping[val]])
        return np.array(q)

    def parametrize(self, p=None, smirks_map=None, mapping=None):
        if None in (p, mapping):
            (p, mapping) = self.get_force_field_parameters(use_jax=True)
        return self.get_system_parameters(p=p, mapping=mapping, use_jax=True)

    def parametrize_partial(self):
        return partial(self.parametrize, smirks_map=self.smirks_map, mapping=self.get_force_field_parameters()[1])

    def get_param_matrix(self):
        if not jax_available:
            raise JAXNotInstalledError
        import jax
        (p, mapping) = self.get_force_field_parameters(use_jax=True)
        parametrize_partial = partial(self.parametrize, smirks_map=self.smirks_map, mapping=mapping)

        jac_parametrize = jax.jacfwd(parametrize_partial)
        jac_res = jac_parametrize(p)

        return jac_res.reshape(-1, p.flatten().shape[0])




class SMIRNOFFProperTorsionTerm(SMIRNOFFPotentialTerm):

    name: str = 'ProperTorsions'


class SMIRNOFFImproperTorsionTerm(SMIRNOFFPotentialTerm):

    name: str = 'ImproperTorsions'


class ElectrostaticsTerm(SMIRNOFFPotentialTerm):
    # Note: This class is structured differently from other SMIRNOFFPotentialTerm children

    name: str = 'Electrostatics'
    smirks_map: Dict[tuple, str] = dict()
    potentials: Dict[str, unit.Quantity] = dict()

    @classmethod
    def build_from_toolkit_data(cls, topology, forcefield=None):

        term = cls(name='Electrostatics')
        term.smirks_map = dummy_atomic_slots_map(topology=topology)
        term.potentials = build_smirks_potential_map_electrostatics(forcefield=forcefield, topology=topology, smirks_map=term.smirks_map)
        return term


SUPPORTED_HANDLER_MAPPING = {
    'vdW': SMIRNOFFvdWTerm,
    'Bonds': SMIRNOFFBondTerm,
    'Angles': SMIRNOFFAngleTerm,
    'ProperTorsions': SMIRNOFFProperTorsionTerm,
    'ImproperTorsions': SMIRNOFFImproperTorsionTerm,
    'Electrostatics': ElectrostaticsTerm,
}


class SMIRNOFFTermCollection(BaseModel):

    terms: Dict[str, SMIRNOFFPotentialTerm] = dict()

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def from_toolkit_data(cls, toolkit_forcefield, toolkit_topology):
        collection = cls()
        # Temporary workaround for dealing with charge handlers individually
        partial_charges = get_partial_charges_from_openmm_system(
            toolkit_forcefield.create_openmm_system(toolkit_topology)
        )

        # TODO: Properly deal with the case of partial charges intentionally being zeros
        if np.allclose(partial_charges, 0):
            smirks_map = dummy_atomic_slots_map(topology=toolkit_topology)
            potentials = build_smirks_potential_map_electrostatics(
                forcefield=None,
                topology=toolkit_topology,
                smirks_map=smirks_map,
                partial_charges=partial_charges,
            )
        else:
            smirks_map = build_slot_smirks_map_term(
                toolkit_forcefield['Electrostatics'],
                topology=toolkit_topology
            )

            potentials = build_smirks_potential_map_electrostatics(
                forcefield=toolkit_forcefield,
                topology=toolkit_topology,
                smirks_map=smirks_map,
                partial_charges=partial_charges,
            )

        electrostatics_term = ElectrostaticsTerm(
            name='Electrostatics',
            smirks_map=smirks_map,
            potentials=potentials,
        )

        for handler_to_drop in ['Constraints', 'ToolkitAM1BCC', 'Electrostatics']:
            # TODO: toolkit_forcefield.registered_parameter_handlers when OFFTK 0.7.1 is released
            if handler_to_drop in toolkit_forcefield._parameter_handlers.keys():
                toolkit_forcefield._parameter_handlers.pop(handler_to_drop)

        for handler_name, handler in toolkit_forcefield._parameter_handlers.items():
            collection.add_parameter_handler(handler=handler, forcefield=toolkit_forcefield, topology=toolkit_topology)
        return collection

    def add_parameter_handler(self, handler, topology, forcefield=None):
        handler_name = handler._TAGNAME
        if handler_name in SUPPORTED_HANDLER_MAPPING.keys():
            term = SUPPORTED_HANDLER_MAPPING[handler_name]()

            if handler_name == 'Electrostatics':
                self.terms[handler_name] = term.build_from_toolkit_data(
                    topology=topology,
                    forcefield=forcefield,
                )
            else:
                self.terms[handler_name] = term.build_from_toolkit_data(
                    handler=handler,
                    topology=topology,
                    forcefield=forcefield,
                )
        else:
            raise SMIRNOFFHandlerNotImplementedError(handler_name)
