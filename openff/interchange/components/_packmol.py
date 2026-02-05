"""
A wrapper around PACKMOL. Adapted from OpenFF Evaluator v0.4.3.
"""

from openff.packmol import RHOMBIC_DODECAHEDRON, UNIT_CUBE
from openff.packmol._packmol import (
    _find_packmol,
    _check_add_positive_mass,
    _check_box_shape_shape,
    _validate_inputs,
    _box_vectors_are_in_reduced_form,
    _unit_vec,
    _compute_brick_from_box_vectors,
    _range_neg_pos,
    _iter_lattice_vecs,
    _wrap_into,
    _wrap_into_brick,
    _box_from_density,
    _scale_box,
    _create_solute_pdb,
    _create_molecule_pdbs,
    _get_packmol_version,
    _build_input_file,
    _center_topology_at,
    _max_dist_between_points,
    _load_positions,
)
from openff.packmol import (
    RHOMBIC_DODECAHEDRON_XYHEX,
    pack_box,
    solvate_topology,
    solvate_topology_nonwater,
)

__all__ = (
    "pack_box",
    "solvate_topology",
    "solvate_topology_nonwater",
    "UNIT_CUBE",
    "RHOMBIC_DODECAHEDRON",
    "RHOMBIC_DODECAHEDRON_XYHEX",
    "_find_packmol",
    "_check_add_positive_mass",
    "_check_box_shape_shape",
    "_validate_inputs",
    "_box_vectors_are_in_reduced_form",
    "_unit_vec",
    "_compute_brick_from_box_vectors",
    "_range_neg_pos",
    "_iter_lattice_vecs",
    "_wrap_into",
    "_wrap_into_brick",
    "_box_from_density",
    "_scale_box",
    "_create_solute_pdb",
    "_create_molecule_pdbs",
    "_get_packmol_version",
    "_build_input_file",
    "_center_topology_at",
    "_max_dist_between_points",
    "_load_positions",
)
