"""
Common helpers for exporting virtual sites.
"""
from collections import defaultdict

import numpy
from openff.units import Quantity, unit

from openff.interchange import Interchange
from openff.interchange.exceptions import (
    MissingPositionsError,
    MissingVirtualSitesError,
    VirtualSiteTypeNotImplementedError,
)
from openff.interchange.models import VirtualSiteKey


def _virtual_site_parent_molecule_mapping(
    interchange: Interchange,
) -> dict[VirtualSiteKey, int]:
    mapping: dict[VirtualSiteKey, int] = dict()

    if "VirtualSites" not in interchange.collections:
        return mapping

    for virtual_site_key in interchange["VirtualSites"].key_map:
        assert isinstance(virtual_site_key, VirtualSiteKey)
        parent_atom_index = virtual_site_key.orientation_atom_indices[0]
        parent_atom = interchange.topology.atom(parent_atom_index)
        parent_molecule = parent_atom.molecule
        mapping[virtual_site_key] = interchange.topology.molecule_index(parent_molecule)

    return mapping


def get_positions_with_virtual_sites(
    interchange: Interchange,
    use_zeros: bool = False,
) -> Quantity:
    """Return the positions of all particles (atoms and virtual sites)."""
    if interchange.positions is None:
        raise MissingPositionsError(
            f"Positions are required, found {interchange.positions=}.",
        )

    if "VirtualSites" not in interchange.collections:
        raise MissingVirtualSitesError()

    if len(interchange["VirtualSites"].key_map) == 0:
        raise MissingVirtualSitesError()

    molecule_virtual_site_map = defaultdict(list)

    virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

    for virtual_site, molecule_index in virtual_site_molecule_map.items():
        molecule_virtual_site_map[molecule_index].append(virtual_site)

    particle_positions = Quantity(
        numpy.empty(shape=(0, 3)),
        unit.nanometer,
    )

    for molecule in interchange.topology.molecules:
        molecule_index = interchange.topology.molecule_index(molecule)

        atom_indices = [
            interchange.topology.atom_index(atom) for atom in molecule.atoms
        ]
        this_molecule_atom_positions = interchange.positions[atom_indices, :]

        n_virtual_sites_in_this_molecule: int = len(
            molecule_virtual_site_map[molecule_index],
        )

        if use_zeros:
            this_molecule_virtual_site_positions = Quantity(
                numpy.zeros((n_virtual_sites_in_this_molecule, 3)),
                unit.nanometer,
            )

        else:
            this_molecule_virtual_site_positions = Quantity(
                numpy.asarray(
                    [
                        _get_virtual_site_positions(
                            virtual_site_key,
                            interchange,
                        ).m_as(unit.nanometer)
                        for virtual_site_key in molecule_virtual_site_map[
                            molecule_index
                        ]
                    ],
                ),
                unit.nanometer,
            )

        particle_positions = numpy.concatenate(
            [
                particle_positions,
                this_molecule_atom_positions,
                this_molecule_virtual_site_positions,
            ],
        )

    return particle_positions


def _get_virtual_site_positions(
    virtual_site_key: VirtualSiteKey,
    interchange: Interchange,
) -> Quantity:
    # TODO: Move this behavior elsewhere, possibly to a non-GROMACS location
    if virtual_site_key.type == "BondCharge":
        return _get_bond_charge_virtual_site_positions(virtual_site_key, interchange)
    elif virtual_site_key.type == "DivalentLonePair":
        return _get_divalent_lone_pair_virtual_site_positions(
            virtual_site_key,
            interchange,
        )
    elif virtual_site_key.type == "TrivalentLonePair":
        return _get_trivalent_lone_pair_virtual_site_positions(
            virtual_site_key,
            interchange,
        )
    else:
        raise VirtualSiteTypeNotImplementedError(
            f"Virtual site type {virtual_site_key.type} not implemented.",
        )


def _get_bond_charge_virtual_site_positions(
    virtual_site_key,
    interchange,
) -> Quantity:
    potential_key = interchange["VirtualSites"].key_map[virtual_site_key]
    potential = interchange["VirtualSites"].potentials[potential_key]
    distance = potential.parameters["distance"]

    # r1 and r2 are positions of atom1 and atom2 in the convention of
    # these diagrams:
    # https://docs.openforcefield.org/projects/toolkit/en/stable/users/virtualsites.html#applying-virtual-site-parameters
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r2 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]

    r2_r1_norm = numpy.linalg.norm((r2 - r1).m_as(unit.angstrom)) * unit.angstrom

    # The virtual site is placed at a distance opposite the r1 -> r2 vector
    return r1 - (r2 - r1) * (distance / r2_r1_norm)


def _get_monovalent_lone_pair_virtual_site_positions(
    virtual_site_key,
    interchange,
) -> Quantity:
    # r1 and r2 are positions of atom1 and atom2 in the convention of
    # these diagrams:
    # https://docs.openforcefield.org/projects/toolkit/en/stable/users/virtualsites.html#applying-virtual-site-parameters
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r2 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]

    potential_key = interchange["VirtualSites"].key_map[virtual_site_key]
    potential = interchange["VirtualSites"].potentials[potential_key]
    distance = potential.parameters["distance"]
    in_plane_angle = potential.parameters["inPlaneAngle"]
    out_of_plane_angle = potential.parameters["outOfPlaneAngle"]

    if out_of_plane_angle.m != 0.0:
        raise NotImplementedError(
            "Only planar `MonovalentLonePairType` is currently supported.",
        )

    if in_plane_angle.m != 0.0:
        raise NotImplementedError(
            "Only linear `MonovalentLonePairType` is currently supported.",
        )

    # r1 and r2 are positions of atom1 and atom2 in the convention of
    # these diagrams:
    # https://docs.openforcefield.org/projects/toolkit/en/stable/users/virtualsites.html#applying-virtual-site-parameters
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r2 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]

    r2_r1_norm = numpy.linalg.norm((r2 - r1).m_as(unit.angstrom)) * unit.angstrom

    # The virtual site is placed at a distance opposite the r1 -> r2 vector
    return r1 - (r2 - r1) * (distance / r2_r1_norm)


def _get_divalent_lone_pair_virtual_site_positions(
    virtual_site_key,
    interchange,
) -> Quantity:
    r0 = interchange.positions[virtual_site_key.orientation_atom_indices[0]]
    r1 = interchange.positions[virtual_site_key.orientation_atom_indices[1]]
    r2 = interchange.positions[virtual_site_key.orientation_atom_indices[2]]

    r0_r1_bond_length = numpy.sqrt(numpy.sum((r0 - r1) ** 2))
    r0_r2_bond_length = numpy.sqrt(numpy.sum((r0 - r2) ** 2))
    rmid = (r1 + r2) / 2
    rmid_distance = numpy.sqrt(numpy.sum((r0 - rmid) ** 2))

    if abs(r0_r1_bond_length - r0_r2_bond_length) > Quantity(1e-3, unit.nanometer):
        raise VirtualSiteTypeNotImplementedError(
            "Only symmetric geometries (i.e. r2 - r0 = r1 - r0) are currently supported",
        )

    potential_key = interchange["VirtualSites"].key_map[virtual_site_key]
    potential = interchange["VirtualSites"].potentials[potential_key]
    distance = potential.parameters["distance"]
    out_of_plane_angle = potential.parameters["outOfPlaneAngle"]

    if out_of_plane_angle.m_as(unit.degree) != 0.0:
        raise VirtualSiteTypeNotImplementedError(
            "Only planar `DivalentLonePairType` is currently supported.",
        )

    return r0 + (r0 - rmid) * (distance) / (rmid_distance)


def _get_trivalent_lone_pair_virtual_site_positions(
    virtual_site_key,
    interchange,
):
    potential_key = interchange["VirtualSites"].key_map[virtual_site_key]
    distance = (
        interchange["VirtualSites"]
        .potentials[potential_key]
        .parameters["distance"]
        .m_as(unit.nanometer)
    )

    center, a, b, c = (
        interchange.positions[index].m_as(unit.nanometer)
        for index in virtual_site_key.orientation_atom_indices
    )

    # clockwise vs. counter-clockwise matters here - manually correcting it later
    dir = numpy.cross(b - a, c - a)
    dir /= numpy.linalg.norm(dir)

    # if adding the (normalized) normal vector to the midpoint of (a, b, c) ends up closer to the enter
    if numpy.linalg.norm(
        center - (numpy.cross(b - a, c - a) + dir),
    ) < numpy.linalg.norm(center - (numpy.cross(b - a, c - a) - dir)):
        # then this vector is pointing _toward_ the central atom
        pass
    else:
        # otherwise, this vector is pointing _away_ from the central atom, so we need to point it the other way
        dir *= -1

    return Quantity(center + dir * distance, unit.nanometer)
