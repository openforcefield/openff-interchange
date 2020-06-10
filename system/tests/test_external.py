from simtk.openmm.app import PDBFile

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.topology import Topology, Molecule
from openforcefield.utils import get_data_file_path

from ..system import System
from ..utils import get_test_file_path
from .base_test import BaseTest


class TestFromOpenMM(BaseTest):

    def test_from_openmm_pdbfile(self, argon_top, argon_ff):
        # TODO: Host files like this here instead of grabbing from the toolkit
        pdb_file_path = get_test_file_path('10-argons.pdb')
        pdbfile = PDBFile(pdb_file_path.as_posix())

        System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=pdbfile.positions,
            box=pdbfile.topology.getPeriodicBoxVectors(),
        )
