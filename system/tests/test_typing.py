from ..typing.smirnoff import build_smirnoff_map, build_smirnoff_collection
from .base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):
    def test_typing(self, argon_ff, argon_top):
        smirks_map = build_smirnoff_map(forcefield=argon_ff, topology=argon_top)

        assert all([smirks == '[#18:1]' for slot, smirks in smirks_map['vdW'].items()])

    # There's a better way to this; pytest doesn't let fixtures be passed to parametrize
    # TODO: check for proper conversion, not just completeness
    def test_smirnoff_collection(self, argon_ff, ammonia_ff):
        smirnoff_collection = build_smirnoff_collection(forcefield=argon_ff)

        assert sorted(smirnoff_collection.handlers.keys()) == ['vdW']

        smirnoff_collection = build_smirnoff_collection(forcefield=ammonia_ff)

        assert sorted(smirnoff_collection.handlers.keys()) == ['Angles', 'Bonds', 'vdW']
