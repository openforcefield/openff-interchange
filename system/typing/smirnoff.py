from warnings import warn

from ..collections import PotentialHandler, PotentialCollection
from ..utils import simtk_to_pint
from ..potential import ParametrizedAnalyticalPotential as Potential


# TODO: Probably shouldn't have this as a global variable floating around
SUPPORTED_HANDLERS = {'vdW', 'Bonds', 'Angles'}


def build_smirnoff_map(topology, forcefield):
    """Turn a SMIRNOFF force field into a mapping between slots and SMIRKS"""
    typing_map = dict()

    for handler in forcefield._parameter_handlers.keys():

        slot_potential_map = dict()

        matches = forcefield.get_parameter_handler(handler).find_matches(topology)

        for atom_key, atom_match in matches.items():
            slot_potential_map[atom_key] = atom_match.parameter_type.smirks

        typing_map[handler] = slot_potential_map

    return typing_map


def add_handler(forcefield, potential_collection, handler_name):
    """Temporary stand-in for .to_potential calls in toolkit ParameterHandler objects."""
    if handler_name not in SUPPORTED_HANDLERS:
        warn(f'handler {handler_name} not implemented')
        return potential_collection

    if handler_name == 'vdW':
        potential_collection.handlers.update({'vdW': PotentialHandler(name='vdW')})
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

            potential_collection.handlers['vdW'].potentials[param.smirks] = potential

    elif handler_name == 'Bonds':
        potential_collection.handlers.update({'Bonds': PotentialHandler(name='Bonds')})
        for param in forcefield.get_parameter_handler('Bonds').parameters:
            k = simtk_to_pint(param.k)
            length = simtk_to_pint(param.length)

            potential = Potential(
                name=param.id,
                smirks=param.smirks,
                expression='1/2*k*(length-length_0)**2',
                independent_variables={'length_0'},
                parameters={'k': k, 'length': length},
            )

            potential_collection.handlers['Bonds'].potentials[param.smirks] = potential

    elif handler_name == 'Angles':
        potential_collection.handlers.update({'Angles': PotentialHandler(name='Angles')})
        for param in forcefield.get_parameter_handler('Angles').parameters:
            k = simtk_to_pint(param.k)
            angle = simtk_to_pint(param.angle)

            potential = Potential(
                name=param.id,
                smirks=param.smirks,
                expression='1/2*k*(angle-angle_0)**2',
                independent_variables={'angle_0'},
                parameters={'k': k, 'angle': angle},
            )

            potential_collection.handlers['Angles'].potentials[param.smirks] = potential

    return potential_collection


def build_smirnoff_collection(forcefield):
    """Build a PotentialCollection storing data in a SMIRNOFF force field."""
    handlers = [h for h in forcefield._parameter_handlers.keys() if h in SUPPORTED_HANDLERS]

    smirnoff_collection = PotentialCollection()

    for handler in handlers:
        smirnoff_collection = add_handler(forcefield, smirnoff_collection, handler)

    return smirnoff_collection


class SMIRNOFFPotentialHandler(PotentialHandler):

    pass


class SMIRNOFFCollection(PotentialCollection):

    @classmethod
    def from_toolkit_forcefield(cls, toolkit_forcefield):

        return build_smirnoff_collection(toolkit_forcefield)
