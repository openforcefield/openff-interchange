import numpy as np
import pytest
from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.exceptions import MissingPositionsError
from openff.system.stubs import ForceField


def test_nonbonded_compatibility():
    mol = Molecule.from_smiles("CCO")
    mol.name = "FOO"
    mol.generate_conformers(n_conformers=1)

    top = mol.to_topology()
    positions = mol.conformers[0].in_units_of(omm_unit.nanometer) / omm_unit.nanometer
    box = [4, 4, 4] * np.eye(3)

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    off_sys.box = box

    off_sys.handlers["Electrostatics"].method = "reaction-field"

    with pytest.raises(
        NotImplementedError, match="Electrostatics method not supported"
    ):
        off_sys.to_openmm()

    off_sys.handlers["Electrostatics"].method = "PME"

    with pytest.raises(MissingPositionsError):
        off_sys.to_gro("out.gro", writer="internal")

    off_sys.positions = positions

    # Ensure nothing obvious was mangled
    off_sys.to_top("out.top", writer="internal")
    off_sys.to_gro("out.gro", writer="internal")
