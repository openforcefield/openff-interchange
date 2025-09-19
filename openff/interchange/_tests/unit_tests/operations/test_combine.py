import copy

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.exceptions import (
    CutoffMismatchError,
    SwitchingFunctionMismatchError,
    UnsupportedCombinationError,
)
from openff.interchange.warnings import InterchangeCombinationWarning


class TestCombine:
    @skip_if_missing("openmm")
    def test_basic_combination(self, sage_unconstrained):
        """Test basic use of Interchange.__add__() based on the README example"""
        top = MoleculeWithConformer.from_smiles("C").to_topology()

        interchange = Interchange.from_smirnoff(sage_unconstrained, top)

        interchange.box = [4, 4, 4] * numpy.eye(3)

        # Copy and translate atoms by [1, 1, 1]
        other = Interchange(topology=copy.deepcopy(interchange.topology))
        other = copy.deepcopy(interchange)
        other.positions += Quantity("1.0 nanometer")

        combined = interchange.combine(other)

        # Just see if it can be converted into OpenMM and run
        get_openmm_energies(combined)

    @pytest.mark.filterwarnings("ignore:Setting positions to None")
    @pytest.mark.slow
    def test_parameters_do_not_clash(self, monkeypatch, sage_unconstrained):
        thf = MoleculeWithConformer.from_smiles("C1CCOC1")
        ace = MoleculeWithConformer.from_smiles("CC(=O)O")

        def make_interchange(molecule: MoleculeWithConformer) -> Interchange:
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

    def test_positions_setting(self, sage):
        """Test that positions exist on the result if and only if
        both input objects have positions."""

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
        sage,
        basic_top,
        handler,
    ):
        sage_modified = ForceField("openff-2.1.0.offxml")

        sage_modified[handler].cutoff *= 1.5

        with pytest.raises(
            CutoffMismatchError,
            match=f"{handler} cutoffs do not match",
        ):
            sage.create_interchange(basic_top).combine(
                sage_modified.create_interchange(basic_top),
            )

    @pytest.mark.parametrize("key", ["Electrostatics", "vdW"])
    def test_cutoffs_only_slightly_differ(
        self,
        sage,
        basic_top,
        key,
    ):
        """Test that, by default, combining proceeds if 1-16 < cutoff_diff < 1e-6."""
        int1 = sage.create_interchange(basic_top)
        int2 = int1.__deepcopy__()

        # should combine with a tiny difference that's still fails x1 == x2, but ...
        int2[key].cutoff = int1[key].cutoff + Quantity("1e-10 nanometer")

        int1.combine(int2)

        # ... fails when the difference is more than 1e-6
        int2[key].cutoff = int1[key].cutoff + Quantity("1e-4 nanometer")

        with pytest.raises(
            CutoffMismatchError,
        ):
            int1.combine(int2)

    def test_error_mismatched_switching_function(
        self,
        sage,
        basic_top,
    ):
        sage_modified = ForceField("openff-2.1.0.offxml")

        sage_modified["vdW"].switch_width *= 0.0

        with pytest.raises(
            SwitchingFunctionMismatchError,
        ):
            sage.create_interchange(basic_top).combine(
                sage_modified.create_interchange(basic_top),
            )

    @pytest.mark.parametrize(
        argnames=["vdw", "electrostatics"],
        argvalues=[
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    def test_dont_combine_mixed_14(self, sage, vdw, electrostatics):
        """
        Until it's implemented, error out when any non-bonded collections have non-equal 1-4 scaling factors.

        See https://github.com/openforcefield/openff-interchange/issues/380

        """
        interchange1 = sage.create_interchange(MoleculeWithConformer.from_smiles("C").to_topology())
        interchange2 = sage.create_interchange(MoleculeWithConformer.from_smiles("CCO").to_topology())

        if vdw:
            interchange2["vdW"].scale_14 = 0.444

        if electrostatics:
            interchange2["Electrostatics"].scale_14 = 0.444

        if vdw or electrostatics:
            with pytest.raises(
                UnsupportedCombinationError,
                match="1-4.*vdW" if vdw else "1-4.*Electro",
            ):
                interchange2.combine(interchange1)
        else:
            # if neither are modified, that error shouldn't be raised
            interchange2.combine(interchange1)

    def test_mix_different_5_6_rounding(self, parsley, sage, ethanol):
        """Test that 0.833333 is 'upconverted' to 0.8333333333 in combination."""
        with pytest.warns(
            InterchangeCombinationWarning,
            match="more digits in rounding",
        ):
            parsley.create_interchange(
                ethanol.to_topology(),
            ).combine(
                sage.create_interchange(ethanol.to_topology()),
            )

    @pytest.mark.parametrize("flip_order", [False, True])
    def test_no_unnecessary_duplicate_tags(self, water_dimer, tip3p, sage, ethanol, flip_order):
        """Check that the '_DUPLICATE' hack is only used when necessary. See Issue #1324."""
        interchanges = [
            tip3p.create_interchange(
                water_dimer,
            ),
            sage.create_interchange(
                ethanol.to_topology(),
            ),
        ]

        if flip_order:
            interchanges.reverse()

        combined = interchanges[0].combine(interchanges[1])

        for collection_name in ["vdW", "Bonds", "Angles", "ProperTorsions", "Constraints"]:
            for potential_key in combined[collection_name].key_map.values():
                assert "_DUPLICATE" not in potential_key.id, (
                    f"Failed sanity check with {potential_key.id} in {collection_name}"
                )

    @pytest.mark.parametrize("flip_order", [False, True])
    @pytest.mark.parametrize("handler_name", ["Constraints", "Bonds", "Angles", "ProperTorsions"])
    def test_constraint_key_collision(self, parsley, sage, ethanol, flip_order, handler_name):
        """Test that key collisions in constraints and valence terms are handled."""
        interchanges = [
            parsley.create_interchange(
                ethanol.to_topology(),
            ),
            sage.create_interchange(
                ethanol.to_topology(),
            ),
        ]

        # want to make sure this behavior is not order-dependent
        if flip_order:
            interchanges.reverse()

        arrays_before = [interchange[handler_name].get_system_parameters() for interchange in interchanges]

        numpy.testing.assert_raises(AssertionError, numpy.testing.assert_allclose, *arrays_before)

        array_after_combine = interchanges[0].combine(interchanges[1])[handler_name].get_system_parameters()

        # check that the contents of the combined Interchange contains each input, without mushing
        numpy.testing.assert_allclose(
            numpy.vstack(arrays_before),
            array_after_combine,
        )

    def test_DUPLICATE_key_already_exists(self, methane):
        import numpy as np

        # use unconstrained FFs to ensure bond parameters aren't overwritten by constraints
        ff1 = ForceField("openff_unconstrained-1.2.1.offxml")
        ff2 = ForceField("openff_unconstrained-2.2.1.offxml")
        ic1 = ff1.create_interchange(methane.to_topology())
        ic2 = ff2.create_interchange(methane.to_topology())

        # Manually compile an array of expected final system params
        individual_params = np.concatenate(
            [
                ic1["Bonds"].get_system_parameters(),
                ic2["Bonds"].get_system_parameters(),
                ic1["Bonds"].get_system_parameters(),
            ],
            axis=0,
        )

        # Get the array of system params from and interchange resulting from successive .combine operations
        ic3 = ic1.combine(ic2.combine(ic1))
        combined_params = ic3["Bonds"].get_system_parameters()

        # Test the arrays for equality
        for idx, (p1, p2) in enumerate(zip(individual_params, combined_params)):
            assert np.allclose(p1, p2), f"mismatch at bond {idx=}"
