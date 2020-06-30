import pytest
import numpy as np

from openforcefield.topology import Molecule, Topology

from ..typing.smirnoff import SMIRNOFFPotentialTerm, SMIRNOFFvdWTerm, SMIRNOFFTermCollection, ElectrostaticsTerm, SUPPORTED_HANDLERS
from ..exceptions import SMIRNOFFHandlerNotImplementedError
from ..utils import get_partial_charges_from_openmm_system
from .base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):

    # There's probably a better way to this, but pytest doesn't let fixtures be passed to parametrize
    # TODO: check for proper conversion, not just completeness
    def test_reconstruct_toolkit_forcefield(self, argon_ff, argon_top, ammonia_ff, ammonia_top):
        ff_collection = SMIRNOFFTermCollection.from_toolkit_data(argon_ff, argon_top)

        assert [*ff_collection.terms.keys()] == ['vdW']

        assert all([smirks == '[#18:1]' for smirks in ff_collection.terms['vdW'].smirks_map.values()])

        ff_collection = SMIRNOFFTermCollection.from_toolkit_data(ammonia_ff, ammonia_top)

        assert sorted(ff_collection.terms.keys()) == ['Angles', 'Bonds', 'vdW']

        for term in ff_collection.terms.values():
            assert term.potentials.keys() is not None

    def test_more_map_functions(self, parsley, cyclohexane_top):
        # TODO: Better way of testing individual handlers

        term_collection = SMIRNOFFTermCollection()

        # TODO: This should just be `term_collection = SMIRNOFFTermCollection.from_toolkit_data(parsley, cyclohexane_top)`
        for name, handler in parsley._parameter_handlers.items():
            if name in SUPPORTED_HANDLERS:
                term_collection.add_parameter_handler(handler=handler, topology=cyclohexane_top, forcefield=parsley)

        # Do this in separate steps so that Electrostatics handler can access the handlers it depends on
        handlers_to_drop = []
        for name in parsley._parameter_handlers.keys():
            if name not in SUPPORTED_HANDLERS:
                handlers_to_drop.append(name)

        for name in handlers_to_drop:
            parsley._parameter_handlers.pop(name)

        assert sorted(term_collection.terms.keys()) == sorted(SUPPORTED_HANDLERS)

    def test_construct_term_from_toolkit_forcefield(self, parsley, ethanol_top):
        val1 = SMIRNOFFPotentialTerm.build_from_toolkit_data(handler=parsley['vdW'], topology=ethanol_top)
        val2 = SMIRNOFFvdWTerm.build_from_toolkit_data(handler=parsley['vdW'], topology=ethanol_top)

        # TODO: DO something here that isn't so dangerous
        assert val1 == val2

        ref = get_partial_charges_from_openmm_system(parsley.create_openmm_system(ethanol_top))

        eh = ElectrostaticsTerm.build_from_toolkit_data(handler=parsley['Electrostatics'], forcefield=parsley, topology=ethanol_top)
        partial_charges = [*eh.potentials.values()]

        assert np.allclose(partial_charges, ref)

    def test_unimplemented_conversions(self, parsley, ethanol_top):

        # TODO: Replace this with a system contained a truly unsupported potential
        with pytest.raises(SMIRNOFFHandlerNotImplementedError):
            SMIRNOFFTermCollection.from_toolkit_data(parsley, ethanol_top)
