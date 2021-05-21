import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.utilities.testing import skip_if_missing

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

    @needs_gmx
    @needs_lmp
    @pytest.mark.slow
    @skip_if_missing("foyer")
    def test_atom_ordering(self):
        """Test that atom indices in bonds are ordered consistently between the slot map and topology"""
        import foyer

        from openff.system.components.foyer import from_foyer
        from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
        from openff.system.tests.energy_tests.lammps import get_lammps_energies
        from openff.system.tests.energy_tests.openmm import get_openmm_energies

        oplsaa = foyer.forcefields.load_OPLSAA()

        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "BENZ"
        biotop = OFFBioTop.from_molecules(benzene)
        biotop.mdtop = md.Topology.from_openmm(biotop.to_openmm())
        out = from_foyer(ff=oplsaa, topology=biotop)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # Violates OPLS-AA, but the point here is just to make sure everything runs
        out["vdW"].mixing_rule = "lorentz-berthelot"

        get_gromacs_energies(out)
        get_openmm_energies(out)
        get_lammps_energies(out)
