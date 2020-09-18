import numpy as np
import parmed as pmd
from openforcefield.typing.engines.smirnoff import ForceField
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

    def test_basic_conversion_argon(self, argon_ff, argon_top):
        struct = System.from_toolkit(
            forcefield=argon_ff, topology=argon_top
        ).to_parmed()
        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

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
