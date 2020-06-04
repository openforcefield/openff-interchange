from ..collections import PotentialHandler, PotentialCollection
from ..utils import simtk_to_pint
from ..potential import ParametrizedAnalyticalPotential as Potential


def build_smirnoff_map(topology, forcefield):
    """Turn a SMIRNOFF force field into a mapping between slots and SMIRKS"""
    typing_map = dict()

    for handler_name in ['vdW']:
        slot_potential_map = dict()

        matches = forcefield.get_parameter_handler(handler_name).find_matches(topology)

        for atom_key, atom_match in matches.items():
            slot_potential_map[atom_key] = atom_match.parameter_type.smirks

        typing_map[handler_name] = slot_potential_map

    return typing_map


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
                'vdW': SMIRNOFFPotentialHandler(
                    name='vdW',
                    potentials={
                        param.smirks: potential,
                    }
                )

            }

    return potential_collection

class SMIRNOFFPotentialHandler(PotentialHandler):

    pass

class SMIRNOFFCollection(PotentialCollection):

    @classmethod
    def from_toolkit_forcefield(cls, toolkit_forcefield):

        toolkit_handlers = toolkit_forcefield._parameter_handlers.keys()
        supported_handlers = ['vdW']

        for handler in toolkit_handlers:
            if handler not in supported_handlers:
                continue
            handler_conversion(toolkit_forcefield, cls, handler)
        return cls
