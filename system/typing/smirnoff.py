from warnings import warn

from ..collections import PotentialHandler, PotentialCollection
from ..utils import simtk_to_pint
from ..potential import ParametrizedAnalyticalPotential as Potential


def build_smirnoff_map(topology, forcefield):
    """Turn a SMIRNOFF force field into a mapping between slots and SMIRKS"""
    typing_map = dict()

    if 'vdW' in forcefield._parameter_handlers.keys():
        slot_potential_map = dict()

        matches = forcefield.get_parameter_handler('vdW').find_matches(topology)

        for atom_key, atom_match in matches.items():
            slot_potential_map[atom_key] = atom_match.parameter_type.smirks

        typing_map['vdW'] = slot_potential_map

    if 'Bonds' in forcefield._parameter_handlers.keys():
        slot_potential_map = dict()

        matches = forcefield.get_parameter_handler('Bonds').find_matches(topology)

        for bond_key, bond_match in matches.items():
            slot_potential_map[bond_key] = bond_match.parameter_type.smirks

        typing_map['Bonds'] = slot_potential_map

    return typing_map


def add_handler(forcefield, potential_collection, handler_name):
    """Temporary stand-in for .to_potential calls in toolkit ParameterHandler objects."""
    if handler_name not in ['vdW', 'Bonds']:
        warn(f'handler {handler_name} not implemented')

    if handler_name == 'vdW':
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
                    'vdW': SMIRNOFFPotentialHandler(
                        name='vdW',
                        potentials={
                            param.smirks: potential,
                        }
                    )
    
                }

    elif handler_name == 'Bonds':
        for param in forcefield.get_parameter_handler('Bonds').parameters:
            k = simtk_to_pint(param.k)
            length = simtk_to_pint(param.length)

            potential = Potential(
                name=param.id,
                smirks=param.smirks,
                expression='1/2*k(length-length_0)**2',
                independent_variables={'length_0'},
                parameters={'k': k, 'length_0': length},
            )

            try:
                potential_collection.handlers['Bonds'][param.smirks] = potential
            except (AttributeError, TypeError):
                potential_collection.handlers = {
                    'Bonds': SMIRNOFFPotentialHandler(
                        name='Bonds',
                        potentials={
                            param.smirks: potential,
                        }
                    )

                }

        return potential_collection


def build_smirnoff_collection(forcefield):
    """Build a PotentialCollection storing data in a SMIRNOFF force field."""
    handlers = [*forcefield._parameter_handlers.keys()]

    smirnoff_collection = PotentialCollection()

    for handler in handlers:
        add_handler(forcefield, smirnoff_collection, handler)

    return smirnoff_collection


class SMIRNOFFPotentialHandler(PotentialHandler):

    pass


class SMIRNOFFCollection(PotentialCollection):

    @classmethod
    def from_toolkit_forcefield(cls, toolkit_forcefield):

        return build_smirnoff_collection(toolkit_forcefield)
