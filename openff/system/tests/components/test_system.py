from copy import deepcopy

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, ParameterHandler
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from pydantic import ValidationError

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.system import System
from openff.system.drivers import get_openmm_energies
from openff.system.exceptions import SMIRNOFFHandlersNotImplementedError
from openff.system.tests import BaseTest
from openff.system.tests.energy_tests.test_energies import needs_gmx, needs_lmp
from openff.system.utils import get_test_file_path


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


def test_box_setter():
    tmp = System()

    with pytest.raises(ValidationError):
        tmp.box = [2, 2, 3, 90, 90, 90]


def test_unimplemented_smirnoff_handler():
    top = Molecule.from_smiles("CC").to_topology()
    parsley = ForceField("openff-1.0.0.offxml")

    bogus_parameter_handler = ParameterHandler(version=0.3)
    bogus_parameter_handler._TAGNAME = "bogus"
    parsley.register_parameter_handler(bogus_parameter_handler)
    with pytest.raises(SMIRNOFFHandlersNotImplementedError, match="SMIRNOFF.*bogus"):
        System.from_smirnoff(force_field=parsley, topology=top)


@pytest.mark.slow
class TestSystemCombination(BaseTest):
    def test_basic_combination(self):
        """Test basic use of System.__add__() based on the README example"""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = OFFBioTop.from_molecules([mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        parsley = ForceField("openff_unconstrained-1.0.0.offxml")
        openff_sys = parsley.create_openff_system(top)

        openff_sys.box = [4, 4, 4] * np.eye(3)
        openff_sys.positions = mol.conformers[0]._value / 10.0

        # Copy and translate atoms by [1, 1, 1]
        other = System()
        other._inner_data = deepcopy(openff_sys._inner_data)
        other.positions += 1.0 * unit.nanometer

        combined = openff_sys + other

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)


class TestSystem(BaseTest):
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

        from openff.system.components.system import System
        from openff.system.drivers import (
            get_gromacs_energies,
            get_lammps_energies,
            get_openmm_energies,
        )

        oplsaa = foyer.forcefields.load_OPLSAA()

        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "BENZ"
        biotop = OFFBioTop.from_molecules(benzene)
        biotop.mdtop = md.Topology.from_openmm(biotop.to_openmm())
        out = System.from_foyer(ff=oplsaa, topology=biotop)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # Violates OPLS-AA, but the point here is just to make sure everything runs
        out["vdW"].mixing_rule = "lorentz-berthelot"

        get_gromacs_energies(out)
        get_openmm_energies(out)
        get_lammps_energies(out)
