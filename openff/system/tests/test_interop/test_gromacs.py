import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule
from openff.utilities.testing import skip_if_missing

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.exceptions import UnsupportedExportError
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.test_energies import needs_gmx
from openff.system.utils import get_test_file_path


@needs_gmx
class TestGROMACS(BaseTest):
    @skip_if_missing("parmed")
    def test_set_mixing_rule(self, ethanol_top, parsley):
        import parmed as pmd

        openff_sys = System.from_smirnoff(force_field=parsley, topology=ethanol_top)

        openff_sys.to_top("lorentz.top")
        top_file = pmd.load_file("lorentz.top")
        assert top_file.combining_rule == "lorentz"

        openff_sys["vdW"].mixing_rule = "geometric"

        openff_sys.to_top("geometric.top")
        import os

        os.system("cat geometric.top")
        top_file = pmd.load_file("geometric.top")
        assert top_file.combining_rule == "geometric"

    @pytest.mark.xfail(
        reason="cannot test unsupported mixing rules in GROMACS with current SMIRNOFFvdWHandler model"
    )
    def test_unsupported_mixing_rule(self, ethanol_top, parsley):
        # TODO: Update this test when the model supports more mixing rules than GROMACS does
        openff_sys = System.from_smirnoff(force_field=parsley, topology=ethanol_top)
        openff_sys["vdW"].mixing_rule = "kong"

        with pytest.raises(UnsupportedExportError, match="rule `geometric` not compat"):
            openff_sys.to_top("out.top")

    def test_residue_names_in_gro_file(self):
        """Test that residue names > 5 characters don't break .gro file output"""
        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "supercalifragilisticexpialidocious"
        top = OFFBioTop.from_molecules(benzene)
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        # Populate an entire system because ...
        force_field = ForceField("openff-1.0.0.offxml")
        out = force_field.create_openff_system(top)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # ... the easiest way to check the validity of the files
        # is to see if GROMACS can run them
        get_gromacs_energies(out)
