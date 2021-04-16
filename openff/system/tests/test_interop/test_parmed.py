import numpy as np
import pytest
from openff.units import unit

from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.utils import top_from_smiles


class TestParmedConversion(BaseTest):
    @pytest.fixture()
    def box(self):
        return np.array([4.0, 4.0, 4.0])

    def test_box(self, argon_ff, argon_top, box):
        off_sys = argon_ff.create_openff_system(topology=argon_top, box=box)
        off_sys.positions = (
            np.zeros(shape=(argon_top.n_topology_atoms, 3)) * unit.angstrom
        )
        struct = off_sys._to_parmed()

        assert np.allclose(
            struct.box[:3],
            [40, 40, 40],
        )

    def test_basic_conversion_argon(self, argon_ff, argon_top, box):
        off_sys = argon_ff.create_openff_system(argon_top, box=box)
        off_sys.positions = np.zeros(shape=(argon_top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    def test_basic_conversion_params(self, box):
        top = top_from_smiles("C")
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

        off_sys = parsley.create_openff_system(topology=top, box=box)
        # UnitArray(...)
        off_sys.positions = np.zeros(shape=(top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        sigma0 = struct.atoms[0].atom_type.sigma
        epsilon0 = struct.atoms[0].atom_type.epsilon

        sigma1 = struct.atoms[1].atom_type.sigma
        epsilon1 = struct.atoms[1].atom_type.epsilon

        bond_k = struct.bonds[0].type.k
        req = struct.bonds[0].type.req

        angle_k = struct.angles[0].type.k
        theteq = struct.angles[0].type.theteq

        assert sigma0 == pytest.approx(3.3996695084235347)
        assert epsilon0 == pytest.approx(0.1094)

        assert sigma1 == pytest.approx(2.649532787749369)
        assert epsilon1 == pytest.approx(0.0157)

        assert bond_k == pytest.approx(379.04658864565)
        assert req == pytest.approx(1.092888378383)

        assert angle_k == pytest.approx(37.143507635885)
        assert theteq == pytest.approx(107.5991506326)

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    def test_basic_conversion_ammonia(self, ammonia_ff, ammonia_top, box):
        off_sys = ammonia_ff.create_openff_system(ammonia_top, box=box)
        off_sys.positions = np.zeros(shape=(ammonia_top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))
