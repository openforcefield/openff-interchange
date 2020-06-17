import pytest

from ..typing.smirnoff import SMIRNOFFPotentialTerm, SMIRNOFFvdWTerm, SMIRNOFFTermCollection
from ..exceptions import SMIRNOFFHandlerNotImplementedError
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

    def test_construct_term_from_toolkit_forcefield(self, argon_ff, argon_top):
        val1 = SMIRNOFFPotentialTerm.build_from_toolkit_data(name='vdW', forcefield=argon_ff, topology=argon_top)
        val2 = SMIRNOFFvdWTerm.build_from_toolkit_data(name='vdW', forcefield=argon_ff, topology=argon_top)

        # TODO: DO something here that isn't so dangerous
        assert val1 == val2

    def test_unimplemented_conversions(self, parsley, ethanol_top):

        # TODO: Replace this with a system contained a truly unsupported potential
        with pytest.raises(SMIRNOFFHandlerNotImplementedError):
            SMIRNOFFTermCollection.from_toolkit_data(parsley, ethanol_top)
