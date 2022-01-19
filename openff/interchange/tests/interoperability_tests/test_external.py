import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import get_data_file_path
from openff.units import unit
from openff.units.openmm import from_openmm
from openmm import app
from openmm.unit import nanometer as nm

from openff.interchange import Interchange
from openff.interchange.components.mdtraj import _OFFBioTop
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.drivers.openmm import _get_openmm_energies
from openff.interchange.tests import _BaseTest, get_test_file_path


class TestFromOpenMM(_BaseTest):
    @pytest.mark.slow()
    def test_from_openmm_pdbfile(self, argon_ff, argon_top):
        pdb_file_path = get_test_file_path("10-argons.pdb")
        pdbfile = app.PDBFile(pdb_file_path)

        mol = Molecule.from_smiles("[#18]")
        top = _OFFBioTop.from_openmm(pdbfile.topology, unique_molecules=[mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())
        box = pdbfile.topology.getPeriodicBoxVectors()
        box = box.value_in_unit(nm) * unit.nanometer

        out = Interchange.from_smirnoff(argon_ff, top)
        out.box = box
        out.positions = from_openmm(pdbfile.getPositions())

        assert np.allclose(
            out.positions.to(unit.nanometer).magnitude,
            pdbfile.getPositions().value_in_unit(nm),
        )

        get_openmm_energies(out, hard_cutoff=True).compare(
            _get_openmm_energies(
                omm_sys=argon_ff.create_openmm_system(top),
                box_vectors=pdbfile.topology.getPeriodicBoxVectors(),
                positions=pdbfile.getPositions(),
                hard_cutoff=True,
            )
        )

    @pytest.fixture()
    def unique_molecules(self):
        molecules = ["O", "C1CCCCC1", "C", "CCC", "CCO", "CCCCO"]
        return [Molecule.from_smiles(mol) for mol in molecules]
        # What if, instead ...
        # Molecule.from_iupac(molecules)

    @pytest.mark.skip(
        reason="Needs to be reimplmented after OFFTK 0.11.0 with fewer moving parts"
    )
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "pdb_path",
        [
            ("cyclohexane_ethanol_0.4_0.6.pdb"),
            ("cyclohexane_water.pdb"),
            ("ethanol_water.pdb"),
            ("propane_methane_butanol_0.2_0.3_0.5.pdb"),
        ],
    )
    def test_from_toolkit_packmol_boxes(self, parsley, pdb_path, unique_molecules):
        """
        Test loading some pre-prepared PACKMOL-generated systems.

        These use PDB files already prepared in the toolkit because PDB files are a pain.
        """
        pdb_file_path = get_data_file_path("systems/packmol_boxes/" + pdb_path)
        pdbfile = app.PDBFile(pdb_file_path)
        top = _OFFBioTop.from_openmm(
            pdbfile.topology,
            unique_molecules=unique_molecules,
        )
        top.mdtop = md.Topology.from_openmm(top.to_openmm())
        box = pdbfile.topology.getPeriodicBoxVectors()
        box = box.value_in_unit(nm) * unit.nanometer

        out = Interchange.from_smirnoff(parsley, top)
        out.box = box
        out.positions = from_openmm(pdbfile.getPositions())

        assert np.allclose(
            out.positions.to(unit.nanometer).magnitude,
            pdbfile.getPositions().value_in_unit(nm),
        )

        get_openmm_energies(
            out,
            hard_cutoff=True,
            combine_nonbonded_forces=True,
        ).compare(
            _get_openmm_energies(
                omm_sys=parsley.create_openmm_system(top),
                box_vectors=pdbfile.topology.getPeriodicBoxVectors(),
                positions=pdbfile.getPositions(),
                hard_cutoff=True,
            )
        )
