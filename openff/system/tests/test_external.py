import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils import get_data_file_path
from simtk import openmm
from simtk.unit import nanometer as nm

from openff.system import unit
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.openmm import (
    _get_openmm_energies,
    get_openmm_energies,
)
from openff.system.utils import get_test_file_path


class TestFromOpenMM(BaseTest):
    def test_from_openmm_pdbfile(self, argon_ff, argon_top):
        pdb_file_path = get_test_file_path("10-argons.pdb")
        pdbfile = openmm.app.PDBFile(pdb_file_path)

        mol = Molecule.from_smiles("[#18]")
        top = Topology.from_openmm(pdbfile.topology, unique_molecules=[mol])
        box = pdbfile.topology.getPeriodicBoxVectors()
        box = box.value_in_unit(nm) * unit.nanometer

        out = argon_ff.create_openff_system(top)
        out.box = box
        out.positions = pdbfile.getPositions()

        assert np.allclose(
            out.positions.to(unit.nanometer).magnitude,
            pdbfile.getPositions().value_in_unit(nm),
        )

        get_openmm_energies(out).compare(
            _get_openmm_energies(
                omm_sys=argon_ff.create_openmm_system(top),
                box_vectors=pdbfile.topology.getPeriodicBoxVectors(),
                positions=pdbfile.getPositions(),
            )
        )

    @pytest.fixture
    def unique_molecules(self):
        molecules = ["O", "C1CCCCC1", "C", "CCC", "CCO", "CCCCO"]
        return [Molecule.from_smiles(mol) for mol in molecules]
        # What if, instead ...
        # Molecule.from_iupac(molecules)

    @pytest.mark.parametrize(
        "pdb_path",
        [
            ("cyclohexane_ethanol_0.4_0.6.pdb"),
            ("cyclohexane_water.pdb"),
            ("ethanol_water.pdb"),
            ("propane_methane_butanol_0.2_0.3_0.5.pdb"),
        ],
    )
    @pytest.mark.skip
    def test_from_toolkit_packmol_boxes(self, pdb_path, unique_molecules):
        """
        Test loading some pre-prepared PACKMOL-generated systems.

        These use PDB files already prepared in the toolkit because PDB files are a pain.
        """
        pdb_file_path = get_data_file_path("systems/packmol_boxes/" + pdb_path)
        pdbfile = openmm.app.PDBFile(pdb_file_path)
        Topology.from_openmm(
            pdbfile.topology,
            unique_molecules=unique_molecules,
        )
        # proto_system = ProtoSystem(
        #     topology=off_top,
        #     positions=pdbfile.positions,
        #     box=pdbfile.topology.getPeriodicBoxVectors(),
        # )

        # assert np.allclose(proto_system.positions, simtk_to_pint(pdbfile.positions))
        # assert np.allclose(
        #     proto_system.box,
        #     simtk_to_pint(pdbfile.topology.getPeriodicBoxVectors()),
        # )
