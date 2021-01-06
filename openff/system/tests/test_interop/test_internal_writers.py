import numpy as np
from openforcefield.topology import Molecule

from openff.system.stubs import ForceField


def test_internal_gro_writer():
    top = Molecule.from_smiles("C").to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(top)

    out.box = [4, 4, 4] * np.eye(3)
    out.positions = np.random.rand(15).reshape((5, 3))

    out.to_gro("internal.gro", writer="internal")
    out.to_gro("parmed.gro", writer="parmed")

    with open("internal.gro", "r") as file1:
        with open("parmed.gro", "r") as file2:
            # Ignore first two lines and last line
            assert file1.readlines()[2:-1] == file2.readlines()[2:-1]
