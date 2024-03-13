import numpy as np
import parmed
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.utilities import (
    get_data_file_path,
    has_executable,
    has_package,
    skip_if_missing,
)

from openff.interchange import Interchange
from openff.interchange._tests import get_test_file_path, requires_openeye
from openff.interchange.drivers import get_amber_energies, get_openmm_energies
from openff.interchange.exceptions import UnsupportedExportError

if has_package("openmm"):
    import openmm
    import openmm.app
    import openmm.unit


@pytest.mark.slow()
@requires_openeye
@pytest.mark.parametrize(
    "molecule",
    [
        "lig_CHEMBL3265016-1.pdb",
        "c1ccc2ccccc2c1",
    ],
)
def test_atom_names_with_padding(molecule):
    # pytest processes fixtures before the decorator can be applied
    if molecule.endswith(".pdb"):
        molecule = Molecule(get_test_file_path(molecule).as_posix())
    else:
        molecule = Molecule.from_smiles(molecule)

    # Unclear if the toolkit will always load PDBs with padded whitespace in name
    Interchange.from_smirnoff(
        ForceField("openff-2.0.0.offxml"),
        molecule.to_topology(),
    ).to_prmtop("tmp.prmtop")

    # Loading with ParmEd striggers #679 if exclusions lists are wrong
    parmed.load_file("tmp.prmtop")


@pytest.mark.parametrize("molecule", ["C1=CN=CN1", "c1ccccc1", "c1ccc2ccccc2c1"])
def exclusions_in_rings(molecule):
    molecule.generate_conformers(n_conformers=1)
    topology = molecule.to_topology()
    topology.box_vectors = [4, 4, 4]

    sage_no_impropers = ForceField("openff-2.0.0.offxml")
    sage_no_impropers.deregister_parameter_handler("ImproperTorsions")

    interchange = sage_no_impropers.create_interchange(topology)

    interchange.to_prmtop("tmp.prmtop")

    # Use the OpenMM export as source of truth
    openmm_system = interchange.to_openmm()
    for force in openmm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            reference = force.getNumExceptions()

    loaded_system = openmm.app.AmberPrmtopFile("tmp.prmtop").createSystem()
    for force in loaded_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            assert force.getNumExceptions() == reference


def test_virtual_site_error(tip4p, water):
    with pytest.raises(
        UnsupportedExportError,
        match="not yet supported in Amber writers",
    ):
        tip4p.create_interchange(water.to_topology()).to_prmtop("foo")


class TestAmber:
    @pytest.mark.skip(reason="Need replacement route to reference positions")
    def test_inpcrd(self, sage):
        mol = Molecule.from_smiles(10 * "C")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(force_field=sage, topology=mol.to_topology())
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.positions = unit.nanometer * np.round(out.positions.m_as(unit.nanometer), 5)

        out.to_inpcrd("internal.inpcrd")
        # This method no longer exists
        out._to_parmed().save("parmed.inpcrd")

        coords1 = parmed.load_file("internal.inpcrd").coordinates
        coords2 = parmed.load_file("parmed.inpcrd").coordinates

        np.testing.assert_equal(coords1, coords2)

    @skip_if_missing("openmm")
    @pytest.mark.skipif(not has_executable("sander"), reason="sander not installed")
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "smiles",
        [
            "C",
            "CC",  # Adds a proper torsion term(s)
            "C=O",  # Simplest molecule with any improper torsion
            "OC=O",  # Simplest molecule with a multi-term torsion
            "CCOC",  # This hits t86, which has a non-1.0 idivf
            "C1COC(=O)O1",  # This adds an improper, i2
        ],
    )
    def test_amber_energy(self, sage_unconstrained, smiles):
        """
        Basic test to see if the amber energy driver is functional.

        Note this test can only use the unconstrained version of Sage because sander applies SHAKE
        constraints in the single-point energy calculation, i.e. uses geometries with constraints
        applied, NOT what is in the coordinate file. See issue
        https://github.com/openforcefield/openff-interchange/issues/323
        """
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)

        interchange.box = [5, 5, 5]
        interchange.positions = mol.conformers[0]

        omm_energies = get_openmm_energies(interchange, combine_nonbonded_forces=True)
        amb_energies = get_amber_energies(interchange)

        # TODO: More investigation into possible non-bonded energy differences and better reporting.
        #       03/02/2023 manually inspected some files and charges and vdW parameters are
        #       precisely identical. Passing box vectors to prmtop files might not always work.
        omm_energies.energies.pop("Nonbonded")
        amb_energies.energies.pop("vdW")
        amb_energies.energies.pop("Electrostatics")

        omm_energies.compare(amb_energies)


class TestPRMTOP:
    @skip_if_missing("openmm")
    @pytest.mark.slow()
    def test_atom_names_pdb(self):
        peptide = Molecule.from_polymer_pdb(
            get_data_file_path("proteins/MainChain_ALA_ALA.pdb", "openff.toolkit"),
        )
        ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml")

        Interchange.from_smirnoff(ff14sb, peptide.to_topology()).to_prmtop(
            "atom_names.prmtop",
        )

        pdb_object = openmm.app.PDBFile(
            get_data_file_path("proteins/MainChain_ALA_ALA.pdb", "openff.toolkit"),
        )
        openmm_object = openmm.app.AmberPrmtopFile("atom_names.prmtop")

        pdb_atom_names = [atom.name for atom in pdb_object.topology.atoms()]

        openmm_atom_names = [atom.name for atom in openmm_object.topology.atoms()]

        assert openmm_atom_names == pdb_atom_names


class TestAmberResidues:
    @pytest.mark.parametrize("patch_residue_name", [True, False])
    def test_single_residue_system_residue_name(
        self,
        tmp_path,
        sage,
        ethanol,
        patch_residue_name,
    ):
        if patch_residue_name:
            for atom in ethanol.atoms:
                atom.metadata["residue_name"] = "YUP"

            ethanol.add_default_hierarchy_schemes()

        Interchange.from_smirnoff(sage, [ethanol]).to_prmtop("test.prmtop")

        residue_names = [r.name for r in parmed.load_file("test.prmtop").residues]

        if patch_residue_name:
            assert residue_names == ["YUP"]
        else:
            assert residue_names == ["RES"]
