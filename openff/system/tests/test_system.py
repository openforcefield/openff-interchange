import numpy as np
import pytest
from openff.toolkit.topology import Molecule

from openff.system.stubs import ForceField
from openff.system.units import unit


def test_getitem():
    """Test behavior of System.__getitem__"""
    mol = Molecule.from_smiles("CCO")
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(mol.to_topology())

    out.box = [4, 4, 4]

    assert not out.positions
    np.testing.assert_equal(out["box"].m, (4 * np.eye(3) * unit.nanometer).m)
    np.testing.assert_equal(out["box"].m, out["box_vectors"].m)

    assert out["Bonds"] == out.handlers["Bonds"]

    with pytest.raises(LookupError, match="Only str"):
        out[1]

    with pytest.raises(LookupError, match="Could not find"):
        out["CMAPs"]
