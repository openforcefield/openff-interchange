import copy

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Topology, unit
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.exceptions import (
    CutoffMismatchError,
    SwitchingFunctionMismatchError,
)


class TestCombine:
    @skip_if_missing("openmm")
    def test_basic_combination(self, monkeypatch, sage_unconstrained):
        """Test basic use of Interchange.__add__() based on the README example"""
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        mol = Molecule.from_smiles("C")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol])

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)

        interchange.box = [4, 4, 4] * numpy.eye(3)
        interchange.positions = mol.conformers[0]

        # Copy and translate atoms by [1, 1, 1]
        other = Interchange()
        other = copy.deepcopy(interchange)
        other.positions += 1.0 * unit.nanometer

        combined = interchange.combine(other)

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)

    def test_parameters_do_not_clash(self, monkeypatch, sage_unconstrained):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        thf = Molecule.from_smiles("C1CCOC1")
        ace = Molecule.from_smiles("CC(=O)O")

        thf.generate_conformers(n_conformers=1)
        ace.generate_conformers(n_conformers=1)

        def make_interchange(molecule: Molecule) -> Interchange:
            molecule.generate_conformers(n_conformers=1)
            interchange = Interchange.from_smirnoff(
                force_field=sage_unconstrained,
                topology=[molecule],
            )
            interchange.positions = molecule.conformers[0]

            return interchange

        thf_interchange = make_interchange(thf)
        ace_interchange = make_interchange(ace)
        complex_interchange = thf_interchange.combine(ace_interchange)

        thf_vdw = thf_interchange["vdW"].get_system_parameters()
        ace_vdw = ace_interchange["vdW"].get_system_parameters()
        add_vdw = complex_interchange["vdW"].get_system_parameters()

        numpy.testing.assert_equal(numpy.vstack([thf_vdw, ace_vdw]), add_vdw)

        # TODO: Ensure the de-duplication is maintained after exports

    def test_positions_setting(self, monkeypatch, sage):
        """Test that positions exist on the result if and only if
        both input objects have positions."""

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        ethane = Molecule.from_smiles("CC")
        methane = Molecule.from_smiles("C")

        ethane_interchange = Interchange.from_smirnoff(
            sage,
            [ethane],
        )
        methane_interchange = Interchange.from_smirnoff(sage, [methane])

        ethane.generate_conformers(n_conformers=1)
        methane.generate_conformers(n_conformers=1)

        assert (methane_interchange.combine(ethane_interchange)).positions is None
        methane_interchange.positions = methane.conformers[0]
        assert (methane_interchange.combine(ethane_interchange)).positions is None
        ethane_interchange.positions = ethane.conformers[0]
        assert (methane_interchange.combine(ethane_interchange)).positions is not None

    @pytest.mark.parametrize("handler", ["Electrostatics", "vdW"])
    def test_error_mismatched_cutoffs(
        self,
        monkeypatch,
        sage,
        basic_top,
        handler,
    ):

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        sage_modified = ForceField("openff-2.1.0.offxml")

        sage_modified[handler].cutoff *= 1.5

        with pytest.raises(
            CutoffMismatchError,
            match=f"{handler} cutoffs do not match",
        ):
            sage.create_interchange(basic_top).combine(
                sage_modified.create_interchange(basic_top),
            )

    def test_error_mismatched_switching_function(
        self,
        monkeypatch,
        sage,
        basic_top,
    ):

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        sage_modified = ForceField("openff-2.1.0.offxml")

        sage_modified["vdW"].switch_width *= 0.0

        with pytest.raises(
            SwitchingFunctionMismatchError,
        ):
            sage.create_interchange(basic_top).combine(
                sage_modified.create_interchange(basic_top),
            )
