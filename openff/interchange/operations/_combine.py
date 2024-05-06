"""The logic behind `Interchange.combine`."""

import copy
import warnings
from typing import TYPE_CHECKING

import numpy

from openff.interchange.components.toolkit import _combine_topologies
from openff.interchange.exceptions import (
    CutoffMismatchError,
    SwitchingFunctionMismatchError,
    UnsupportedCombinationError,
)

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange


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
            "One or more inputs is missing a vdW and/or Electrostatics handler(s).",
        )

    for key in ["vdW", "Electrostatics"]:
        if interchange1[key].cutoff != interchange2[key].cutoff:
            raise CutoffMismatchError(
                f"{key} cutoffs do not match. Found "
                f"{interchange1[key].cutoff} and {interchange2[key].cutoff}.",
            )

    if interchange1["vdW"].switch_width != interchange2["vdW"].switch_width:
        raise SwitchingFunctionMismatchError(
            f"Switching distance(s) do not match. Found "
            f"{interchange1['vdW'].switch_width} and {interchange2['vdW'].switch_width}.",
        )


def _combine(
    input1: "Interchange",
    input2: "Interchange",
) -> "Interchange":

    warnings.warn(
        "Interchange object combination is experimental and likely to produce "
        "strange results. Any workflow using this method is not guaranteed to "
        "be suitable for production. Use with extreme caution and thoroughly "
        "validate results!",
        stacklevel=2,
    )

    result = copy.deepcopy(input1)

    result.topology = _combine_topologies(input1.topology, input2.topology)
    atom_offset = input1.topology.n_atoms

    _check_nonbonded_compatibility(input1, input2)

    # TODO: Test that charge cache is invalidated in each of these cases
    if "Electrostatics" in input1.collections:
        input1["Electrostatics"]._charges = dict()
        input1["Electrostatics"]._charges_cached = False

    if "Electrostatics" in input2.collections:
        input2["Electrostatics"]._charges = dict()
        input2["Electrostatics"]._charges_cached = False

    for handler_name, handler in input2.collections.items():
        # TODO: Actually specify behavior in this case
        try:
            self_handler = result.collections[handler_name]
        except KeyError:
            result.collections[handler_name] = handler
            warnings.warn(
                f"'other' Interchange object has handler with name {handler_name} not "
                f"found in 'self,' but it has now been added.",
                stacklevel=2,
            )
            continue

        for top_key, pot_key in handler.key_map.items():
            _tmp_pot_key = copy.deepcopy(pot_key)
            new_atom_indices = tuple(idx + atom_offset for idx in top_key.atom_indices)
            new_top_key = top_key.__class__(**top_key.dict())
            try:
                new_top_key.atom_indices = new_atom_indices
            except ValueError:
                assert len(new_atom_indices) == 1
                new_top_key.this_atom_index = new_atom_indices[0]
            # If interchange was not created with SMIRNOFF, we need avoid merging potentials with same key
            if pot_key.associated_handler == "ExternalSource":
                _mult = 0
                while _tmp_pot_key in self_handler.potentials:
                    _tmp_pot_key.mult = _mult
                    _mult += 1

            self_handler.key_map.update({new_top_key: _tmp_pot_key})
            if handler_name == "Constraints":
                self_handler.potentials.update(
                    {_tmp_pot_key: handler.potentials[pot_key]},
                )
            else:
                self_handler.potentials.update(
                    {_tmp_pot_key: handler.potentials[pot_key]},
                )

        # Ensure the charge cache is rebuilt
        if handler_name == "Electrostatics":
            self_handler._charges_cached = False
            self_handler._get_charges()

        result.collections[handler_name] = self_handler

    if result.positions is not None and input2.positions is not None:
        result.positions = numpy.vstack([result.positions, input2.positions])
    else:
        warnings.warn(
            "Setting positions to None because one or both objects added together were missing positions.",
        )
        result.positions = None

    if not numpy.all(result.box == result.box):
        raise UnsupportedCombinationError(
            "Combination with unequal box vectors is not curretnly supported",
        )

    return result
