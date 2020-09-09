import numpy as np
from simtk import unit as omm_unit

from openff.system.system import System

from ..base_test import BaseTest


class TestParmedConversion(BaseTest):
    def test_box(self, argon_ff, argon_top):
        struct = System.from_toolkit(
            forcefield=argon_ff, topology=argon_top
        ).to_parmed()
        assert np.allclose(
            struct.box[:3],
            argon_top.box_vectors.value_in_unit(omm_unit.angstrom).diagonal(),
        )

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")
