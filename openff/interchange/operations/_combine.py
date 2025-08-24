"""The logic behind `Interchange.combine`."""

import copy
import warnings
from typing import TYPE_CHECKING

import numpy
from openff.toolkit import Quantity

from openff.interchange.components.toolkit import _combine_topologies
from openff.interchange.exceptions import (
    CutoffMismatchError,
    SwitchingFunctionMismatchError,
    UnsupportedCombinationError,
)
from openff.interchange.warnings import InterchangeCombinationWarning

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange

DEFAULT_CUTOFF_TOLERANCE = Quantity("1e-6 nanometer")


def _are_almost_equal(
    target: Quantity | float,
    reference: Quantity | float,
    tolerance: Quantity | float,
):
    return abs(reference - target) < tolerance


def _check_nonbonded_compatibility(
    interchange1: "Interchange",
    interchange2: "Interchange",
):
    if not (
        "vdW" in interchange1.collections
        and "vdW" in interchange2.collections
        and "Electrostatics" in interchange1.collections
        and "Electrostatics" in interchange2.collections
    ):
        raise UnsupportedCombinationError(
            "One or more inputs is missing a vdW and/or Electrostatics collection(s).",
        )

    for key in ["vdW", "Electrostatics"]:
        if not _are_almost_equal(
            interchange1[key].cutoff,
            interchange2[key].cutoff,
            DEFAULT_CUTOFF_TOLERANCE,
        ):
            raise CutoffMismatchError(
                f"{key} cutoffs do not match. Found {interchange1[key].cutoff} and {interchange2[key].cutoff}.",  # type: ignore[attr-defined]
            )

    if not _are_almost_equal(
        interchange1["vdW"].switch_width,
        interchange2["vdW"].switch_width,
        DEFAULT_CUTOFF_TOLERANCE,
    ):
        raise SwitchingFunctionMismatchError(
            f"Switching distance(s) do not match. Found "
            f"{interchange1['vdW'].switch_width} and {interchange2['vdW'].switch_width}.",
        )

    if not _are_almost_equal(
        interchange1["vdW"].scale_14,
        interchange2["vdW"].scale_14,
        1e-6,
    ):
        raise UnsupportedCombinationError(
            "1-4 scaling factors in vdW handler(s) do not match.",
        )

    if interchange1["Electrostatics"].scale_14 != interchange2["Electrostatics"].scale_14:
        if sorted([interchange1["Electrostatics"].scale_14, interchange2["Electrostatics"].scale_14]) == [
            0.833333,
            0.8333333333,
        ]:
            warnings.warn(
                "Found electrostatics 1-4 scaling factors of 5/6 with slightly different rounding "
                "(0.833333 and 0.8333333333). This likely stems from OpenFF using more digits in rounding 1/1.2. "
                "The value of 0.8333333333 will be used, which may or may not introduce small errors. ",
                InterchangeCombinationWarning,
            )

            interchange1["Electrostatics"].scale_14 = 0.8333333333
            interchange2["Electrostatics"].scale_14 = 0.8333333333

        else:
            raise UnsupportedCombinationError(
                "1-4 scaling factors in Electrostatics handler(s) do not match.",
            )


def _combine(
    input1: "Interchange",
    input2: "Interchange",
) -> "Interchange":
    warnings.warn(
        "Interchange object combination is complex and may produce strange results outside "
        "of use cases it has been tested in. Use with caution and thoroughly validate results!",
        InterchangeCombinationWarning,
    )

    result = copy.deepcopy(input1)

    result._topology = _combine_topologies(input1.topology, input2.topology)
    atom_offset = input1.topology.n_atoms

    _check_nonbonded_compatibility(input1, input2)

    # TODO: Test that charge cache is invalidated in each of these cases
    if "Electrostatics" in input1.collections:
        input1["Electrostatics"]._charges = dict()
        input1["Electrostatics"]._charges_cached = False

    if "Electrostatics" in input2.collections:
        input2["Electrostatics"]._charges = dict()
        input2["Electrostatics"]._charges_cached = False

    for collection_name, collection in input2.collections.items():
        # TODO: Actually specify behavior in this case
        try:
            self_collection = result.collections[collection_name]
        except KeyError:
            result.collections[collection_name] = collection
            warnings.warn(
                f"'other' Interchange object has collection with name {collection_name} not "
                f"found in 'self,' but it has now been added.",
            )
            continue

        for top_key, pot_key in collection.key_map.items():
            _tmp_pot_key = copy.deepcopy(pot_key)
            new_atom_indices = tuple(idx + atom_offset for idx in top_key.atom_indices)
            new_top_key = top_key.__class__(**top_key.model_dump())
            try:
                new_top_key.atom_indices = new_atom_indices  # type: ignore[misc]
            except (ValueError, AttributeError):
                assert len(new_atom_indices) == 1
                new_top_key.this_atom_index = new_atom_indices[0]  # type: ignore
            # If interchange was not created with SMIRNOFF, we need avoid merging potentials with same key
            if pot_key.associated_handler == "ExternalSource":
                _mult = 0
                while _tmp_pot_key in self_collection.potentials:
                    _tmp_pot_key.mult = _mult
                    _mult += 1

            self_collection.key_map.update({new_top_key: _tmp_pot_key})
            if collection_name == "Constraints":
                self_collection.potentials.update(
                    {_tmp_pot_key: collection.potentials[pot_key]},
                )
            else:
                self_collection.potentials.update(
                    {_tmp_pot_key: collection.potentials[pot_key]},
                )

        # Ensure the charge cache is rebuilt
        if collection_name == "Electrostatics":
            self_collection._charges_cached = False  # type: ignore[attr-defined]
            self_collection._get_charges()  # type: ignore[attr-defined]

        result.collections[collection_name] = self_collection

    if result.positions is not None and input2.positions is not None:
        result.positions = numpy.vstack([result.positions, input2.positions])
    else:
        warnings.warn(
            "Setting positions to None because one or both objects added together were missing positions.",
            InterchangeCombinationWarning,
        )
        result.positions = None

    if not numpy.all(result.box == result.box):
        raise UnsupportedCombinationError(
            "Combination with unequal box vectors is not curretnly supported",
        )

    return result
