import mdtraj as md
from openff.toolkit.topology import Molecule

from openff.system.components.misc import OFFBioTop
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.utils import get_test_file_path


def test_residue_names_in_gro_file():
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
