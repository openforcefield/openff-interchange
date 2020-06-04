from system.typing.smirnoff import build_slots_parameter_map
from system.tests.base_test import BaseTest


class TestSMIRNOFFTyping(BaseTest):
    def test_typing(self, argon_ff, argon_top):
        smirks_map = build_slots_parameter_map(forcefield=argon_ff, topology=argon_top)

        assert all([smirks == '[#18:1]' for slot, smirks in smirks_map['vdW'].items()])
