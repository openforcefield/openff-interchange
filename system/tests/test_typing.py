import pytest

from ..typing.smirnoff import SMIRNOFFPotentialTerm, SMIRNOFFvdWTerm, SMIRNOFFTermCollection
from ..exceptions import SMIRNOFFHandlerNotImplementedError
from .base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):

    def test_reconstruct_toolkit_forcefield(self, argon_ff, argon_top):
        ff_collection = SMIRNOFFTermCollection.from_toolkit_data(argon_ff, argon_top)

        # TODO: check for proper conversion, not just completeness

        assert [*ff_collection.terms.keys()] == ['vdW']

        assert all([smirks == '[#18:1]' for smirks in ff_collection.terms['vdW'].smirks_map.values()])

    def test_construct_term_from_toolkit_forcefield(self, argon_ff, argon_top):
        val1 = SMIRNOFFPotentialTerm.build_from_toolkit_data(name='vdW', forcefield=argon_ff, topology=argon_top)
        val2 = SMIRNOFFvdWTerm.build_from_toolkit_data(name='vdW', forcefield=argon_ff, topology=argon_top)

        # TODO: DO something here that isn't so dangerous
        assert val1 == val2

    # There's probably a better way to this, but pytest doesn't let fixtures be passed to parametrize
    def test_unimplementedtest_smirnoff_collection(self, ammonia_ff, ammonia_top):

        with pytest.raises(SMIRNOFFHandlerNotImplementedError):
            SMIRNOFFTermCollection.from_toolkit_data(ammonia_ff, ammonia_top)
