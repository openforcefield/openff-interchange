from openff.toolkit.topology import Molecule, Topology

from openff.system.components.mdtraj import OFFBioTop
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.test_energies import needs_gmx, needs_lmp
from openff.system.tests.utils import compare_charges_omm_off
from openff.system.utils import get_test_file_path


class TestStubs(BaseTest):
    """Test the functionality of the stubs.py module"""

    def test_from_parsley(self):

        force_field = ForceField("openff-1.3.0.offxml")

        top = OFFBioTop.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = force_field.create_openff_system(top)

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()

        assert type(out.topology) == OFFBioTop
        assert type(out.topology) != Topology
        assert isinstance(out.topology, Topology)
