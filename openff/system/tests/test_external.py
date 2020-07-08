import pytest
import numpy as np
from simtk.openmm.app import PDBFile

from openforcefield.topology import Molecule, Topology
from openforcefield.utils import get_data_file_path

from ..system import System, ProtoSystem
from ..utils import get_test_file_path, simtk_to_pint
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

    @pytest.fixture
    def unique_molecules(self):
        molecules = ['O', 'C1CCCCC1', 'C', 'CCC', 'CCO', 'CCCCO']
        return [Molecule.from_smiles(mol) for mol in molecules]
        # What if, instead ...
        # Molecule.from_iupac(molecules)

    @pytest.mark.parametrize(
        'pdb_path',
        [
            ('cyclohexane_ethanol_0.4_0.6.pdb'),
            ('cyclohexane_water.pdb'),
            ('ethanol_water.pdb'),
            ('propane_methane_butanol_0.2_0.3_0.5.pdb'),
        ],
    )
    def test_from_toolkit_packmol_boxes(self, pdb_path, unique_molecules):
        """
        Test loading some pre-prepared PACKMOL-generated systems.

        These use PDB files already prepared in the toolkit because PDB files are a pain.
        """
        pdb_file_path = get_data_file_path('systems/packmol_boxes/' + pdb_path)
        pdbfile = PDBFile(pdb_file_path)
        off_top = Topology.from_openmm(
            pdbfile.topology, unique_molecules=unique_molecules,
        )
        proto_system = ProtoSystem(
            topology=off_top,
            positions=pdbfile.positions,
            box=pdbfile.topology.getPeriodicBoxVectors(),
        )

        assert np.allclose(proto_system.positions, simtk_to_pint(pdbfile.positions))
        assert np.allclose(
            proto_system.box, simtk_to_pint(pdbfile.topology.getPeriodicBoxVectors()),
        )
