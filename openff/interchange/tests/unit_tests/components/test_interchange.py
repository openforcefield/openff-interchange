from copy import deepcopy

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ParameterHandler
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from pydantic import ValidationError

from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.exceptions import (
    MissingParameterHandlerError,
    MissingParametersError,
    MissingPositionsError,
    SMIRNOFFHandlersNotImplementedError,
)
from openff.interchange.tests import _BaseTest, get_test_file_path, needs_gmx, needs_lmp


@pytest.mark.slow()
class TestInterchange(_BaseTest):
    def test_getitem(self, parsley):
        """Test behavior of Interchange.__getitem__"""
        mol = Molecule.from_smiles("CCO")
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

    def test_get_parameters(parsley):
        mol = Molecule.from_smiles("CCO")
        out = Interchange.from_smirnoff(force_field=parsley, topology=mol.to_topology())

        from_interchange = out._get_parameters("Bonds", (0, 4))
        from_handler = out["Bonds"]._get_parameters((0, 4))

        assert "k" in from_interchange.keys()
        assert "length" in from_interchange.keys()
        assert from_interchange == from_handler

        with pytest.raises(MissingParameterHandlerError, match="Foobar"):
            out._get_parameters("Foobar", (0, 1))

        with pytest.raises(MissingParametersError, match=r"atoms \(0, 100\)"):
            out._get_parameters("Bonds", (0, 100))

    def test_box_setter(self):
        tmp = Interchange()

        with pytest.raises(ValidationError):
            tmp.box = [2, 2, 3, 90, 90, 90]

    def test_basic_combination(self, parsley_unconstrained):
        """Test basic use of Interchange.__add__() based on the README example"""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        openff_sys = Interchange.from_smirnoff(parsley_unconstrained, top)

        openff_sys.box = [4, 4, 4] * np.eye(3)
        openff_sys.positions = mol.conformers[0]

        # Copy and translate atoms by [1, 1, 1]
        other = Interchange()
        other._inner_data = deepcopy(openff_sys._inner_data)
        other.positions += 1.0 * unit.nanometer

        combined = openff_sys + other

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)

    def test_parameters_do_not_clash(self, parsley_unconstrained):
        thf = Molecule.from_smiles("C1CCOC1")
        ace = Molecule.from_smiles("CC(=O)O")

        thf.generate_conformers(n_conformers=1)
        ace.generate_conformers(n_conformers=1)

        def make_interchange(molecule: Molecule) -> Interchange:
            molecule.generate_conformers(n_conformers=1)
            interchange = Interchange.from_smirnoff(
                force_field=parsley_unconstrained, topology=molecule.to_topology()
            )
            interchange.positions = molecule.conformers[0]

            return interchange

        thf_interchange = make_interchange(thf)
        ace_interchange = make_interchange(ace)
        complex_interchange = thf_interchange + ace_interchange

        thf_vdw = thf_interchange["vdW"].get_system_parameters()
        ace_vdw = ace_interchange["vdW"].get_system_parameters()
        add_vdw = complex_interchange["vdW"].get_system_parameters()

        np.testing.assert_equal(np.vstack([thf_vdw, ace_vdw]), add_vdw)

        # TODO: Ensure the de-duplication is maintained after exports

    def test_positions_setting(self, parsley):
        """Test that positions exist on the result if and only if
        both input objects have positions."""

        ethane = Molecule.from_smiles("CC")
        ethane.generate_conformers(n_conformers=1)
        methane = Molecule.from_smiles("C")
        methane.generate_conformers(n_conformers=1)

        ethane_interchange = Interchange.from_smirnoff(
            parsley,
            ethane.to_topology(),
        )
        methane_interchange = Interchange.from_smirnoff(
            parsley,
            methane.to_topology(),
        )

        assert not (methane_interchange + ethane_interchange).positions
        methane_interchange.positions = methane.conformers[0]
        assert not (methane_interchange + ethane_interchange).positions
        ethane_interchange.positions = ethane.conformers[0]
        assert (methane_interchange + ethane_interchange).positions is not None

    def test_input_topology_not_modified(self, sage):
        molecule = Molecule.from_smiles("CCO")
        molecule.generate_conformers(n_conformers=1)
        molecule.conformers[0] += 1 * unit.angstrom
        topology = molecule.to_topology()
        original = list(topology.molecules)[0].conformers[0]

        Interchange.from_smirnoff(force_field=sage, topology=topology)
        new = list(topology.molecules)[0].conformers[0]

        assert np.sum(original - new) == pytest.approx(0 * unit.angstrom)

    @needs_gmx
    @needs_lmp
    @pytest.mark.slow()
    @skip_if_missing("foyer")
    def test_atom_ordering(self):
        """Test that atom indices in bonds are ordered consistently between the slot map and topology"""
        import foyer

        from openff.interchange import Interchange
        from openff.interchange.drivers import (
            get_gromacs_energies,
            get_lammps_energies,
            get_openmm_energies,
        )

        oplsaa = foyer.forcefields.load_OPLSAA()

        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "BENZ"
        out = Interchange.from_foyer(force_field=oplsaa, topology=benzene.to_topology())
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # Violates OPLS-AA, but the point here is just to make sure everything runs
        out["vdW"].mixing_rule = "lorentz-berthelot"

        get_gromacs_energies(out)
        get_openmm_energies(out)
        get_lammps_energies(out)


class TestUnimplementedSMIRNOFFCases(_BaseTest):
    def test_bogus_smirnoff_handler(self, parsley):
        top = Molecule.from_smiles("CC").to_topology()

        bogus_parameter_handler = ParameterHandler(version=0.3)
        bogus_parameter_handler._TAGNAME = "bogus"
        parsley.register_parameter_handler(bogus_parameter_handler)
        with pytest.raises(
            SMIRNOFFHandlersNotImplementedError, match="SMIRNOFF.*bogus"
        ):
            Interchange.from_smirnoff(force_field=parsley, topology=top)


class TestBadExports(_BaseTest):
    def test_invalid_topology(self, parsley):
        """Test that InvalidTopologyError is caught when passing an unsupported
        topology type to Interchange.from_smirnoff"""
        top = Molecule.from_smiles("CC").to_topology().to_openmm()
        # Thought this should raise ...
        """
        with pytest.raises(
            InvalidTopologyError, match="Could not process topology argument.*openmm.*"
        ):
        """
        # but Pydantic it pre-emptively raising ValidationError because of the type mismatch
        with pytest.raises(ValidationError):
            Interchange.from_smirnoff(force_field=parsley, topology=top)

    def test_gro_file_no_positions(self):
        no_positions = Interchange()
        with pytest.raises(MissingPositionsError, match="Positions are req"):
            no_positions.to_gro("foo.gro")

    def test_gro_file_all_zero_positions(self, parsley):
        top = Topology.from_molecules(Molecule.from_smiles("CC"))
        zero_positions = Interchange.from_smirnoff(force_field=parsley, topology=top)
        zero_positions.positions = np.zeros((top.n_atoms, 3)) * unit.nanometer
        with pytest.warns(UserWarning, match="seem to all be zero"):
            zero_positions.to_gro("foo.gro")
