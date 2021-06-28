from copy import deepcopy

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.tests.utils import get_data_file_path
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, ParameterHandler
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from pydantic import ValidationError

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import OFFBioTop
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.exceptions import (
    InvalidTopologyError,
    MissingPositionsError,
    SMIRNOFFHandlersNotImplementedError,
    SMIRNOFFParameterAttributeNotImplementedError,
)
from openff.interchange.tests import BaseTest
from openff.interchange.tests.energy_tests.test_energies import needs_gmx, needs_lmp
from openff.interchange.utils import get_test_file_path


def test_getitem():
    """Test behavior of Interchange.__getitem__"""
    mol = Molecule.from_smiles("CCO")
    parsley = ForceField("openff-1.0.0.offxml")
    out = Interchange.from_smirnoff(force_field=parsley, topology=mol.to_topology())

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
    tmp = Interchange()

    with pytest.raises(ValidationError):
        tmp.box = [2, 2, 3, 90, 90, 90]


@pytest.mark.slow()
class TestInterchangeCombination(BaseTest):
    def test_basic_combination(self, parsley_unconstrained):
        """Test basic use of Interchange.__add__() based on the README example"""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = OFFBioTop.from_molecules([mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        openff_sys = Interchange.from_smirnoff(parsley_unconstrained, top)

        openff_sys.box = [4, 4, 4] * np.eye(3)
        openff_sys.positions = mol.conformers[0]._value / 10.0

        # Copy and translate atoms by [1, 1, 1]
        other = Interchange()
        other._inner_data = deepcopy(openff_sys._inner_data)
        other.positions += 1.0 * unit.nanometer

        combined = openff_sys + other

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)


class TestUnimplementedSMIRNOFFCases(BaseTest):
    def test_bogus_smirnoff_handler(self, parsley):
        top = Molecule.from_smiles("CC").to_topology()

        bogus_parameter_handler = ParameterHandler(version=0.3)
        bogus_parameter_handler._TAGNAME = "bogus"
        parsley.register_parameter_handler(bogus_parameter_handler)
        with pytest.raises(
            SMIRNOFFHandlersNotImplementedError, match="SMIRNOFF.*bogus"
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=top)

    def test_catch_bond_order_interpolation_bonds(self):
        from openff.toolkit.tests.test_forcefield import xml_ff_bo

        forcefield = ForceField(
            get_data_file_path("test_forcefields/test_forcefield.offxml"),
            xml_ff_bo,
        )

        top = Molecule.from_smiles("CCO").to_topology()

        with pytest.raises(
            SMIRNOFFParameterAttributeNotImplementedError, match="length_bondorder"
        ):
            Interchange.from_smirnoff(force_field=forcefield, topology=top)

    def test_catch_bond_order_interpolation_torsions(self):
        from openff.toolkit.tests.test_forcefield import (
            xml_ff_torsion_bo_standard_supersede,
        )

        forcefield = ForceField(
            get_data_file_path("test_forcefields/test_forcefield.offxml"),
            xml_ff_torsion_bo_standard_supersede,
        )

        top = Molecule.from_smiles("CCO").to_topology()
        with pytest.raises(
            SMIRNOFFParameterAttributeNotImplementedError, match="k.*_bondorder"
        ):
            Interchange.from_smirnoff(force_field=forcefield, topology=top)

    def test_catch_virtual_sites(self):
        from openff.toolkit.tests.test_forcefield import TestForceFieldVirtualSites

        forcefield = ForceField(
            get_data_file_path("test_forcefields/test_forcefield.offxml"),
            TestForceFieldVirtualSites.xml_ff_virtual_sites_monovalent_match_once,
        )

        top = Molecule.from_smiles("CCO").to_topology()

        with pytest.raises(SMIRNOFFHandlersNotImplementedError, match="VirtualSites"):
            Interchange.from_smirnoff(force_field=forcefield, topology=top)


class TestBadExports(BaseTest):
    def test_invalid_topology(self, parsley):
        """Test that InvalidTopologyError is caught when passing an unsupported
        topology type to Interchange.from_smirnoff"""
        top = Molecule.from_smiles("CC").to_topology().to_openmm()
        with pytest.raises(
            InvalidTopologyError, match="Could not process topology argument.*openmm.*"
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=top)

    def test_gro_file_no_positions(self):
        no_positions = Interchange()
        with pytest.raises(MissingPositionsError, match="Positions are req"):
            no_positions.to_gro("foo.gro")

    def test_gro_file_all_zero_positions(self, parsley):
        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        zero_positions = Interchange.from_smirnoff(force_field=parsley, topology=top)
        zero_positions.positions = np.zeros((top.n_topology_atoms, 3)) * unit.nanometer
        with pytest.warns(UserWarning, match="seem to all be zero"):
            zero_positions.to_gro("foo.gro")


class TestInterchange(BaseTest):
    def test_from_parsley(self):

        force_field = ForceField("openff-1.3.0.offxml")

        top = OFFBioTop.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = Interchange.from_smirnoff(force_field, top)

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
    @pytest.mark.slow()
    @skip_if_missing("foyer")
    def test_atom_ordering(self):
        """Test that atom indices in bonds are ordered consistently between the slot map and topology"""
        import foyer

        from openff.interchange.components.interchange import Interchange
        from openff.interchange.drivers import (
            get_gromacs_energies,
            get_lammps_energies,
            get_openmm_energies,
        )

        oplsaa = foyer.forcefields.load_OPLSAA()

        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "BENZ"
        biotop = OFFBioTop.from_molecules(benzene)
        biotop.mdtop = md.Topology.from_openmm(biotop.to_openmm())
        out = Interchange.from_foyer(ff=oplsaa, topology=biotop)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # Violates OPLS-AA, but the point here is just to make sure everything runs
        out["vdW"].mixing_rule = "lorentz-berthelot"

        get_gromacs_energies(out)
        get_openmm_energies(out)
        get_lammps_energies(out)
