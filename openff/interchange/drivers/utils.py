"""Assorted utilities in pre-processing for energy drivers."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange


def _infer_constraints(interchange: "Interchange") -> str:
    if "Constraints" not in interchange.handlers:
        return "none"
    elif "Bonds" not in interchange.handlers:
        return "none"
    else:
        num_constraints = len(interchange["Constraints"].slot_map)
        if num_constraints == 0:
            return "none"
        else:
            if hasattr(interchange.topology, "mdtop"):
                from openff.interchange.components.mdtraj import _get_num_h_bonds

                num_h_bonds = _get_num_h_bonds(interchange.topology.mdtop)
            else:
                from openff.interchange.components.toolkit import _get_num_h_bonds

                num_h_bonds = _get_num_h_bonds(interchange.topology)

            num_bonds = len(interchange["Bonds"].slot_map)
            num_angles = len(interchange["Angles"].slot_map)

            if num_constraints == num_h_bonds:
                return "h-bonds"
            elif num_constraints == len(interchange["Bonds"].slot_map):
                return "all-bonds"
            elif num_constraints == (num_bonds + num_angles):
                return "all-angles"

            else:
                raise Exception("Generic failure while inferring constraints")
