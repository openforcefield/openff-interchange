from copy import deepcopy

import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ElectrostaticsHandler,
    ParameterHandler,
)
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
from openff.interchange.tests import (
    _BaseTest,
    _top_from_smiles,
    get_test_file_path,
    needs_gmx,
    needs_lmp,
)


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

    def test_get_parameters(self, parsley):
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

    def test_missing_electrostatics_handler(self, tip3p_missing_electrostatics_xml):
        """Test that an error is raised when an electrostatics handler is missing"""
        molecule = Molecule.from_smiles("O")
        topology = Topology.from_molecules(molecule)
        topology.box_vectors = unit.Quantity([4, 4, 4], units=unit.nanometer)

        tip3p_missing_electrostatics = ForceField(tip3p_missing_electrostatics_xml)

        with pytest.raises(MissingParameterHandlerError, match="modify partial"):
            Interchange.from_smirnoff(tip3p_missing_electrostatics, topology)

        tip3p = deepcopy(tip3p_missing_electrostatics)

        dummy_electrostatics_handler = ElectrostaticsHandler(skip_version_check=True)
        tip3p.register_parameter_handler(dummy_electrostatics_handler)

        Interchange.from_smirnoff(tip3p, topology)

        tip3p["Electrostatics"].cutoff = 7.89 * unit.angstrom

        out = Interchange.from_smirnoff(tip3p, topology)

        assert out["Electrostatics"].cutoff == 7.89 * unit.angstrom

    def test_box_setter(self):
        tmp = Interchange()

        with pytest.raises(ValidationError):
            tmp.box = [2, 2, 3, 90, 90, 90]

    def test_basic_combination(self, parsley_unconstrained):
        """Test basic use of Interchange.__add__() based on the README example"""
        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol])

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
        methane = Molecule.from_smiles("C")

        ethane_interchange = Interchange.from_smirnoff(
            parsley,
            ethane.to_topology(),
        )
        methane_interchange = Interchange.from_smirnoff(
            parsley,
            methane.to_topology(),
        )

        ethane.generate_conformers(n_conformers=1)
        methane.generate_conformers(n_conformers=1)

        assert (methane_interchange + ethane_interchange).positions is None
        methane_interchange.positions = methane.conformers[0]
        assert (methane_interchange + ethane_interchange).positions is None
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

    def test_from_parsley(self, parsley):

        top = Topology.from_molecules(
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        )

        out = Interchange.from_smirnoff(parsley, top)

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()

        assert type(out.topology) == Topology
        assert isinstance(out.topology, Topology)

    def test_from_parsley_molecule_list(self, parsley):

        out = Interchange.from_smirnoff(
            parsley,
            [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")],
        )

        assert "Constraints" in out.handlers.keys()
        assert "Bonds" in out.handlers.keys()
        assert "Angles" in out.handlers.keys()
        assert "ProperTorsions" in out.handlers.keys()
        assert "vdW" in out.handlers.keys()

        assert type(out.topology) == Topology
        assert isinstance(out.topology, Topology)

    @skip_if_missing("nglview")
    def test_visualize(self, parsley):
        import nglview

        molecule = Molecule.from_smiles("CCO")

        out = Interchange.from_smirnoff(
            force_field=parsley,
            topology=molecule.to_topology(),
        )

        with pytest.raises(
            MissingPositionsError, match="Cannot visualize system without positions."
        ):
            out.visualize()

        molecule.generate_conformers(n_conformers=1)
        out.positions = molecule.conformers[0]

        assert isinstance(out.visualize(), nglview.NGLWidget)


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
        top = _top_from_smiles("CC")
        zero_positions = Interchange.from_smirnoff(force_field=parsley, topology=top)
        zero_positions.positions = np.zeros((top.n_atoms, 3)) * unit.nanometer
        with pytest.warns(UserWarning, match="seem to all be zero"):
            zero_positions.to_gro("foo.gro")
