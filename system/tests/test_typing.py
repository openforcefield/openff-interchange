from ..typing.smirnoff import build_smirnoff_map, build_smirnoff_collection
from .base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):
    def test_typing(self, argon_ff, argon_top):
        smirks_map = build_smirnoff_map(forcefield=argon_ff, topology=argon_top)

        assert all([smirks == '[#18:1]' for slot, smirks in smirks_map['vdW'].items()])

    def test_smirnoff_collection(self, argon_ff):
        smirnoff_collection = build_smirnoff_collection(argon_ff)

        assert all(key == 'vdW' for key in smirnoff_collection.handlers.keys())
