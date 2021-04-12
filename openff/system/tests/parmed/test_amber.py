import numpy as np
from parmed.amber import readparm
from pmdtest.utils import get_fn as get_pmd_fn

from openff.system import unit
from openff.system.components.system import System


class TestParmEdAmber:
    def test_load_prmtop(self):
        struct = readparm.LoadParm(get_pmd_fn("trx.prmtop"))
        other_struct = readparm.AmberParm(get_pmd_fn("trx.prmtop"))
        prmtop = System._from_parmed(struct)
        other_prmtop = System._from_parmed(other_struct)

        for handler_key in prmtop.handlers:
            # TODO: Closer inspection of data
            assert handler_key in other_prmtop.handlers

        assert not prmtop.box

        struct.box = [20, 20, 20, 90, 90, 90]
        prmtop_converted = System._from_parmed(struct)
        np.testing.assert_allclose(
            prmtop_converted.box, np.eye(3) * 2.0 * unit.nanometer
        )

    def test_read_box_parm7(self):
        top = readparm.LoadParm(get_pmd_fn("solv2.parm7"))
        out = System._from_parmed(top)
        # pmd.load_file(get_pmd_fn("solv2.rst7")))
        # top = readparm.LoadParm(get_pmd_fn("solv2.parm7"), xyz=coords.coordinates)
        np.testing.assert_allclose(
            np.diag(out.box.m_as(unit.angstrom)), top.parm_data["BOX_DIMENSIONS"][1:]
        )
