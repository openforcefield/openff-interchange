import numpy as np
import pytest

from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.utils import top_from_smiles
from openff.system.types import UnitArray


class TestParmedConversion(BaseTest):
    @pytest.fixture()
    def box(self):
        return UnitArray(np.array([4, 4, 4]), units="nanometer")

    def test_box(self, argon_ff, argon_top, box):
        sys_out = argon_ff.create_openff_system(topology=argon_top, box=box)
        sys_out.positions = UnitArray(
            np.zeros(
                shape=(argon_top.n_topology_atoms, 3),
            ),
            units="angstrom",
        )
        struct = sys_out.to_parmed()

        assert np.allclose(
            struct.box[:3],
            [40, 40, 40],
        )

    def test_basic_conversion_argon(self, argon_ff, argon_top, box):
        sys_out = argon_ff.create_openff_system(argon_top, box=box)
        sys_out.positions = UnitArray(
            np.zeros(
                shape=(argon_top.n_topology_atoms, 3),
            ),
            units="angstrom",
        )
        struct = sys_out.to_parmed()

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    def test_basic_conversion(self, box):
        top = top_from_smiles("C")
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

        off_sys = parsley.create_openff_system(topology=top, box=box)
        off_sys.positions = UnitArray(
            np.zeros(
                shape=(top.n_topology_atoms, 3),
            ),
            units="angstrom",
        )
        struct = off_sys.to_parmed()

        sigma0 = struct.atoms[0].atom_type.sigma
        epsilon0 = struct.atoms[0].atom_type.epsilon

        sigma1 = struct.atoms[1].atom_type.sigma
        epsilon1 = struct.atoms[1].atom_type.epsilon

        bond_k = struct.bonds[0].type.k
        req = struct.bonds[0].type.req

        angle_k = struct.angles[0].type.k
        theteq = struct.angles[0].type.theteq

        assert sigma0 == pytest.approx(1.6998347542117673)
        assert epsilon0 == pytest.approx(0.1094)

        assert sigma1 == pytest.approx(1.3247663938746845)
        assert epsilon1 == pytest.approx(0.0157)

        assert bond_k == pytest.approx(379.04658864565)
        assert req == pytest.approx(1.092888378383)

        assert angle_k == pytest.approx(37.143507635885)
        assert theteq == pytest.approx(107.5991506326)

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))
