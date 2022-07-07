from typing import TYPE_CHECKING, Dict

import numpy
from openff.toolkit import Molecule
from openff.units import unit

from openff.interchange.models import VirtualSiteKey

if TYPE_CHECKING:
    from openff.interchange import Interchange


def _virtual_site_parent_molecule_mapping(
    interchange: "Interchange",
) -> Dict[VirtualSiteKey, Molecule]:
    mapping = dict()

    for virtual_site_key in interchange["VirtualSites"].slot_map:
        virtual_site_key: VirtualSiteKey  # type: ignore[no-redef]
        parent_atom_index = virtual_site_key.orientation_atom_indices[0]
        parent_atom = interchange.topology.atom(parent_atom_index)
        parent_molecule = parent_atom.molecule
        mapping[virtual_site_key] = parent_molecule

    return mapping


def _get_virtual_site_positions(
    virtual_site_key: VirtualSiteKey, interchange: "Interchange"
) -> unit.Quantity:
    # TODO: Move this behavior elsewhere, possibly to a non-GROMACS location
    if virtual_site_key.type == "BondCharge":
        return _get_bond_charge_virtual_site_positions(virtual_site_key, interchange)
    if virtual_site_key.type == "DivalentLonePair":
        return _get_divalent_lone_pair_virtual_site_positions(
            virtual_site_key, interchange
        )
    else:
        raise Exception(f"Virtual site type {virtual_site_key.type} not implemented.")


def _get_bond_charge_virtual_site_positions(
    virtual_site_key, interchange
) -> unit.Quantity:
    r0 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]

    # numpy.linalg.norm(r0 - r1) requires something unimplemented with __array_function__
    r1_r0_bond_length = numpy.sqrt(numpy.sum((r0 - r1) ** 2))

    potential_key = interchange["VirtualSites"].slot_map[virtual_site_key]
    potential = interchange["VirtualSites"].potentials[potential_key]
    distance = potential.parameters["distance"]

    return r1 + (r1 - r0) * (distance / r1_r0_bond_length)


def _get_divalent_lone_pair_virtual_site_positions(
    virtual_site_key, interchange
) -> unit.Quantity:
    r0 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]
    r2 = interchange.positions[virtual_site_key.orientation_atom_indices[2]]

    r0_r1_bond_length = numpy.sqrt(numpy.sum((r0 - r1) ** 2))
    r0_r2_bond_length = numpy.sqrt(numpy.sum((r0 - r2) ** 2))
    rmid = (r1 + r2) / 2
    rmid_distance = numpy.sqrt(numpy.sum((r0 - rmid) ** 2))

    if abs(r0_r1_bond_length - r0_r2_bond_length) > unit.Quantity(1e-3, unit.nanometer):
        raise Exception(
            "Only symmetric geometries (i.e. r2 - r0 = r1 - r0) are currently supported"
        )

    potential_key = interchange["VirtualSites"].slot_map[virtual_site_key]
    potential = interchange["VirtualSites"].potentials[potential_key]
    distance = potential.parameters["distance"]
    out_of_plane_angle = potential.parameters["outOfPlaneAngle"]

    if out_of_plane_angle.m_as(unit.degree) != 0.0:
        raise Exception("Only planar `DivalentLonePairType` is currently supported.")

    return r0 + (r0 - rmid) * (distance) / (rmid_distance)
