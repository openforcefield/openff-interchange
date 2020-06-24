import numpy as np
from simtk.openmm.app import PDBFile

from openforcefield.topology import Molecule, Topology

from ..system import System, ProtoSystem
from ..utils import get_test_file_path
from .base_test import BaseTest


class TestFromOpenMM(BaseTest):

    def test_from_openmm_pdbfile(self, argon_ff, argon_top):
        # TODO: Host files like this here instead of grabbing from the toolkit
        pdb_file_path = get_test_file_path('10-argons.pdb')
        pdbfile = PDBFile(pdb_file_path)

        argon_system = System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=pdbfile.positions,
            box=pdbfile.topology.getPeriodicBoxVectors(),
        )

        proto_system = ProtoSystem(
            topology=argon_top,
            positions=pdbfile.positions,
            box=pdbfile.topology.getPeriodicBoxVectors(),
        )

        assert np.allclose(argon_system.positions, proto_system.positions)
        assert np.allclose(argon_system.box, proto_system.box)
