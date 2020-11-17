import numpy as np
import parmed as pmd
import pytest
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.tests.base_test import BaseTest
from openff.system.types import UnitArray


class TestParmedConversion(BaseTest):
    def test_box(self, argon_ff, argon_top):
        box = UnitArray(np.array([4, 4, 4]), units="nanometer")
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

    def test_basic_conversion_argon(self, argon_ff, argon_top):
        argon_top.box_vectors = np.array([4, 4, 4]) * omm_unit.nanometer
        sys_out = argon_ff.create_openff_system(argon_top)
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

    @pytest.mark.xfail
    def test_basic_conversion_ethanol(self, ethanol_top):
        # Use the un-constrained Parsley because ParmEd doesn't properly process bond constraints from OpenMM
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")
        struct = System.from_toolkit(
            forcefield=parsley, topology=ethanol_top
        ).to_parmed()

        omm_system = parsley.create_openmm_system(topology=ethanol_top)
        struct_from_omm = pmd.openmm.load_topology(
            topology=ethanol_top.to_openmm(), system=omm_system
        )

        for attr in ["atoms", "bonds", "angles", "dihedrals"]:
            assert len(getattr(struct, attr)) == len(
                getattr(struct_from_omm, attr)
            ), print(attr)
