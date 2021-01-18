import numpy as np
from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.stubs import ForceField


def test_internal_gro_writer():
    mol = Molecule.from_smiles("C")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(top)

    out.box = [4, 4, 4] * np.eye(3)
    out.positions = mol.conformers[0] / omm_unit.nanometer

    out.to_gro("internal.gro", writer="internal")
    out.to_gro("parmed.gro", writer="parmed")

    with open("internal.gro", "r") as file1:
        with open("parmed.gro", "r") as file2:
            # Ignore first two lines and last line
            assert file1.readlines()[2:-1] == file2.readlines()[2:-1]
