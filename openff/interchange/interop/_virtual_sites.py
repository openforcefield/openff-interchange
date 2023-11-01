"""
Common helpers for exporting virtual sites.
"""
from collections import defaultdict
from collections.abc import Iterable
from math import cos, pi, sin
from typing import DefaultDict, Union

import numpy
from openff.units import Quantity, unit

from openff.interchange import Interchange
from openff.interchange.exceptions import (
    MissingPositionsError,
    MissingVirtualSitesError,
    VirtualSiteTypeNotImplementedError,
)
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._virtual_sites import (
    _BondChargeVirtualSite,
    _create_virtual_site_object,
    _DivalentLonePairVirtualSite,
    _MonovalentLonePairVirtualSite,
    _TrivalentLonePairVirtualSite,
    _VirtualSite,
)


def _virtual_site_parent_molecule_mapping(
    interchange: Interchange,
) -> dict[VirtualSiteKey, int]:
    """
    Map `VirtualSiteKey`s the index of the molecule they belong to.

    Parameters
    ----------
    interchange
        The interchange object to get the mapping from.

    Returns
    -------
    mapping: dict[VirtualSiteKey, int]
        A dictionary mapping virtual site keys to the index of the molecule they belong to.

    """
    mapping = dict()

    if "VirtualSites" not in interchange.collections:
        return mapping

    # TODO: This implicitly assumes the ordering of virtual sites is defined by
    # how they are presented in the iterator; this may cause problems when a
    # molecule (a large ligand? polymer?) has many virtual sites
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

    # map of molecule index to *list* of virtual site keys contained therein
    molecule_virtual_site_map: DefaultDict[int, list[VirtualSiteKey]] = defaultdict(
        list,
    )

    virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

    for virtual_site, molecule_index in virtual_site_molecule_map.items():
        molecule_virtual_site_map[molecule_index].append(virtual_site)

    if "VirtualSites" in interchange.collections:
        if use_zeros:
            # TODO: Consider removing this behavior
            virtual_site_positions = numpy.zeros(
                (
                    len(interchange["VirtualSites"].key_map),
                    3,
                ),
            )
        else:
            from openff.interchange.smirnoff._virtual_sites import _generate_positions

            virtual_site_positions = _generate_positions(
                interchange,
                interchange["VirtualSites"],
            )

        return numpy.concatenate(
            [
                interchange.positions,
                virtual_site_positions,
            ],
        )

    else:
        return interchange.positions


def _get_virtual_site_positions(
    virtual_site_key: VirtualSiteKey,
    interchange: Interchange,
) -> Quantity:
    virtual_site_potential = interchange["VirtualSites"].potentials[
        interchange["VirtualSites"].key_map[virtual_site_key]
    ]

    virtual_site: "_VirtualSite" = _create_virtual_site_object(
        virtual_site_key,
        virtual_site_potential,
    )

    try:
        return {
            "BondCharge": _get_bond_charge_virtual_site_positions,
            "MonovalentLonePair": _get_monovalent_lone_pair_virtual_site_positions,
            "DivalentLonePair": _get_divalent_lone_pair_virtual_site_positions,
            "TrivalentLonePair": _get_trivalent_lone_pair_virtual_site_positions,
        }[virtual_site_key.type](virtual_site, interchange)
    except KeyError:
        raise VirtualSiteTypeNotImplementedError(
            f"Virtual site type {virtual_site_key.type} not implemented.",
        )


def _get_bond_charge_weights(
    virtual_site: _BondChargeVirtualSite,
    interchange: Interchange,
) -> tuple[float]:
    """
    Get OpenMM-style weights when using SMIRNOFF `BondCharge`.
    """
    separation = _get_separation_by_atom_indices(
        interchange=interchange,
        atom_indices=virtual_site.orientations,
    )
    distance = virtual_site.distance

    ratio = (distance / separation).m_as(unit.dimensionless)

    w1 = 1.0 + ratio
    w2 = 0.0 - ratio

    return (w1, w2)


def _get_bond_charge_virtual_site_positions(
    virtual_site: _BondChargeVirtualSite,
    interchange: Interchange,
) -> Quantity:
    w1, w2 = _get_bond_charge_weights(
        virtual_site,
        interchange,
    )

    r1 = interchange.positions[virtual_site.orientations[0]]
    r2 = interchange.positions[virtual_site.orientations[1]]

    return r1 * w1 + r2 * w2


def _get_monovalent_weights(
    virtual_site: _MonovalentLonePairVirtualSite,
    interchange: Interchange,
) -> tuple[float]:
    """
    Get OpenMM-style weights when using SMIRNOFF `MonovalentLonePair`.
    """
    if virtual_site.out_of_plane_angle.m != 0.0:
        raise NotImplementedError(
            "Only planar `MonovalentLonePairType` is currently supported."
            f"Given {virtual_site.out_of_plane_angle=}",
        )

    else:
        r12 = _get_separation_by_atom_indices(
            interchange,
            atom_indices=(
                virtual_site.orientations[0],
                virtual_site.orientations[1],
            ),
        )

        r23 = _get_separation_by_atom_indices(
            interchange,
            atom_indices=(
                virtual_site.orientations[1],
                virtual_site.orientations[2],
            ),
        )

        theta = virtual_site.in_plane_angle.m_as(unit.radian)

        theta_123 = _get_angle_by_atom_indices(
            interchange,
            atom_indices=virtual_site.orientations,
        ).m_as(unit.radian)

        w3 = virtual_site.distance / r23 * sin(pi - theta) / sin(pi - theta_123)

        w1 = 1 + w3 * r23 / r12 * cos(pi - theta_123)
        w1 += virtual_site.distance / r12 * cos(pi - theta)

        w2 = 1 - w1 - w3

        return w1, w2, w3


def _get_monovalent_lone_pair_virtual_site_positions(
    virtual_site: _MonovalentLonePairVirtualSite,
    interchange: Interchange,
) -> Quantity:
    w1, w2, w3 = _get_monovalent_weights(
        virtual_site,
        interchange,
    )

    r1 = interchange.positions[virtual_site.orientations[0]]
    r2 = interchange.positions[virtual_site.orientations[1]]
    r3 = interchange.positions[virtual_site.orientations[2]]

    return w1 * r1 + w2 * r2 + w3 * r3


def _get_divalent_weights(
    virtual_site: _DivalentLonePairVirtualSite,
    interchange: Interchange,
) -> Union[tuple[float], tuple[float, float, Quantity]]:
    """
    Get OpenMM-style weights when using SMIRNOFF `DivalentLonePair`.

    If planar, return (w1, w2, w3) for `ThreeParticleAverageSite`.
    If non-planar, return (w12, w13, wcross) for `OutOfPlaneSite`.

    Note that w1, w2, w3, w12, and w13 are unitless but wcross has units
    of inverse distance (1/nm here).
    """
    r12 = _get_separation_by_atom_indices(
        interchange,
        atom_indices=(
            virtual_site.orientations[0],
            virtual_site.orientations[1],
        ),
    )
    r13 = _get_separation_by_atom_indices(
        interchange,
        atom_indices=(
            virtual_site.orientations[0],
            virtual_site.orientations[2],
        ),
    )

    if abs(r12 - r13) > Quantity(1e-3, unit.nanometer):
        raise VirtualSiteTypeNotImplementedError(
            "Only symmetric geometries (i.e. r2 - r0 = r1 - r0) are currently supported",
        )

    theta = _get_angle_by_atom_indices(
        interchange,
        atom_indices=(
            virtual_site.orientations[1],
            virtual_site.orientations[0],
            virtual_site.orientations[2],
        ),
    )

    if virtual_site.out_of_plane_angle.m == 0.0:
        # rmid is a point bisecting hydrogens, also lying on the same line as O-VS
        rmid_distance = r12 * cos(theta.m_as(unit.radian) * 0.5)

        w1 = 1 + float(virtual_site.distance / rmid_distance)
        w2 = w3 = (1 - w1) / 2

        return w1, w2, w3

    else:
        # Special case out 5-site water, assumes symmetric geometry. Other cases
        # fall back to LocalCoordinatesSite implementation
        if sorted(
            [
                interchange.topology.atom(index).atomic_number
                for index in virtual_site.orientations
            ],
        ) != [1, 1, 8]:
            raise VirtualSiteTypeNotImplementedError(
                "Only planar `DivalentLonePairType` is currently supported for non-water.",
            )

        distance_in_plane = virtual_site.distance * numpy.cos(
            virtual_site.out_of_plane_angle.to(unit.radian),
        )
        r1mid = r12 * numpy.cos(theta.m_as(unit.radian) / 2)

        w12 = w13 = -1 * distance_in_plane / r1mid / 2

        # cross product needs positions to determine direction, but the interatomic distances
        # need to be defined by the force field. So use the direction from the positions but
        # distance from force field by scaling the difference in atomic positions by the
        # distance defined by the force field.
        p2 = interchange.positions[virtual_site.orientations[1]]
        p1 = interchange.positions[virtual_site.orientations[0]]
        p3 = interchange.positions[virtual_site.orientations[2]]

        vector_r12 = p2 - p1
        vector_r13 = p3 - p1

        vector_r12 *= r12.m_as(unit.nanometer) / numpy.linalg.norm(
            vector_r12.m_as(unit.nanometer),
        )
        vector_r13 *= r13.m_as(unit.nanometer) / numpy.linalg.norm(
            vector_r13.m_as(unit.nanometer),
        )

        # units of inverse distance
        wcross = virtual_site.distance * numpy.sin(
            virtual_site.out_of_plane_angle.m_as(unit.radian),
        )
        wcross /= Quantity(
            numpy.linalg.norm(numpy.cross(vector_r12, vector_r13).m),
            unit.nanometer**2,
        )

        angle_from_geometry = _get_angle_by_atom_indices(
            interchange,
            atom_indices=(
                virtual_site.orientations[1],
                virtual_site.orientations[0],
                virtual_site.orientations[2],
            ),
            prioritize_geometry=True,
        )

        wcross *= numpy.sin(angle_from_geometry.m_as(unit.radian)) / numpy.sin(
            theta.m_as(unit.radian),
        )

        # wcross /= r12 * r13 * numpy.sin(theta.m_as(unit.radian))

        # print(virtual_site.orientations)
        # arbitrary, should be replaced by a proper cross product
        if virtual_site.orientations[2] > virtual_site.orientations[1]:
            wcross *= -1

        return (
            w12,
            w13,
            wcross,
        )


def _get_divalent_lone_pair_virtual_site_positions(
    virtual_site: _DivalentLonePairVirtualSite,
    interchange: Interchange,
) -> Quantity:
    r1 = interchange.positions[virtual_site.orientations[0]]
    r2 = interchange.positions[virtual_site.orientations[1]]
    r3 = interchange.positions[virtual_site.orientations[2]]

    if virtual_site.out_of_plane_angle.m == 0.0:
        w1, w2, w3 = _get_divalent_weights(
            virtual_site,
            interchange,
        )

        return w1 * r1 + w2 * r2 + w3 * r3

    else:
        w12, w13, wcross = _get_divalent_weights(
            virtual_site,
            interchange,
        )

        # These are based off of the actual positions, not the geometry defined in the force field,
        # though they should be close, so this will not necessarily produce virtual site distances
        # that match what are defined by the force field; this could be changed, however it shouldn't
        # be an issue as long as initial geometries are sane and the actual parameters are written well
        r12 = r2 - r1
        r13 = r3 - r1

        # the sign of the cross product is handled by the geometry
        return r1 + r12 * w12 + r13 * w13 + numpy.abs(wcross) * numpy.cross(r12, r13)


def _get_trivalent_lone_pair_virtual_site_positions(
    virtual_site: _TrivalentLonePairVirtualSite,
    interchange: Interchange,
) -> Quantity:
    distance = virtual_site.distance.m_as(unit.nanometer)

    center, a, b, c = (
        interchange.positions[index].m_as(unit.nanometer)
        for index in virtual_site.orientations
    )

    # clockwise vs. counter-clockwise matters here - manually correcting it later
    dir = numpy.cross(b - a, c - a)
    dir /= numpy.linalg.norm(dir)

    # if adding a differential of the (normalized) normal vector to the midpoint
    # of (a, b, c) ends up closer to the center
    if numpy.linalg.norm(
        center - (numpy.cross(b - a, c - a) + dir * 0.001),
    ) < numpy.linalg.norm(center - (numpy.cross(b - a, c - a) - dir * 0.001)):
        # then this vector is pointing _toward_ the central atom, or "above" in the spec
        pass
    else:
        # otherwise, this vector is pointing _away_ from the central atom, so we need to point it the other way
        dir *= -1

    return Quantity(center + dir * distance, unit.nanometer)


def _get_separation_by_atom_indices(
    interchange: Interchange,
    atom_indices: Iterable[int],
    prioritize_geometry: bool = False,
) -> Quantity:
    """
    Given indices of (two?) atoms, return the distance between them.

    A constraint distance is first searched for, then an equilibrium bond length.

    This is slow, but often necessary for converting virtual site "distances" to weighted
    averages (unitless) of orientation atom positions.
    """
    if prioritize_geometry:
        p1 = interchange.positions[atom_indices[1]]
        p0 = interchange.positions[atom_indices[0]]

        return p1 - p0

    if "Constraints" in interchange.collections:
        collection = interchange["Constraints"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (
                key.atom_indices[::-1] == atom_indices
            ):
                return collection.potentials[collection.key_map[key]].parameters[
                    "distance"
                ]

    if "Bonds" in interchange.collections:
        collection = interchange["Bonds"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (
                key.atom_indices[::-1] == atom_indices
            ):
                return collection.potentials[collection.key_map[key]].parameters[
                    "length"
                ]

    # Two heavy atoms may be on opposite ends of an angle, in which case it's still
    # possible to determine their separation as defined by the geometry of the force field
    if "Angles" in interchange.collections:
        collection = interchange["Angles"]

        index0 = atom_indices[0]
        index1 = atom_indices[1]
        for key in collection.key_map:
            if (key.atom_indices[0] == index0 and key.atom_indices[2] == index1) or (
                key.atom_indices[2] == index0 and key.atom_indices[0] == index1
            ):
                gamma = collection.potentials[collection.key_map[key]].parameters[
                    "angle"
                ]

                a = _get_separation_by_atom_indices(
                    interchange,
                    (key.atom_indices[0], key.atom_indices[1]),
                )

                b = _get_separation_by_atom_indices(
                    interchange,
                    (key.atom_indices[1], key.atom_indices[2]),
                )

                a = a.m_as(unit.nanometer)
                b = b.m_as(unit.nanometer)
                gamma = gamma.m_as(unit.radian)

                # law of cosines
                c2 = a**2 + b**2 - 2 * a * b * numpy.cos(gamma)

                return unit.Quantity(
                    c2**0.5,
                    unit.nanometer,
                )

    raise ValueError(f"Could not find distance between atoms {atom_indices}")


def _get_angle_by_atom_indices(
    interchange: Interchange,
    atom_indices: Iterable[int],
    prioritize_geometry: bool = False,
) -> Quantity:
    """
    Given indices of three atoms, return the angle between them using the law of cosines.

    Distances are defined by the force field, not the positions.

    It is assumed that the second atom is the central atom of the angle.

            b
          /   \
         /     \
        a ----- c

    angle abc = arccos((ac^2 - ab^2 - bc^2) / (-2 * ab * bc)
    """
    if "Angles" in interchange.collections:
        collection = interchange["Angles"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (
                key.atom_indices[::-1] == atom_indices
            ):
                return collection.potentials[collection.key_map[key]].parameters[
                    "angle"
                ]
    else:
        ab = _get_separation_by_atom_indices(
            interchange,
            (atom_indices[0], atom_indices[1]),
            prioritize_geometry=prioritize_geometry,
        ).m_as(unit.nanometer)

        ac = _get_separation_by_atom_indices(
            interchange,
            (atom_indices[0], atom_indices[2]),
            prioritize_geometry=prioritize_geometry,
        ).m_as(unit.nanometer)

        bc = _get_separation_by_atom_indices(
            interchange,
            (atom_indices[1], atom_indices[2]),
            prioritize_geometry=prioritize_geometry,
        ).m_as(unit.nanometer)

        if prioritize_geometry:
            return Quantity(
                numpy.arccos(
                    numpy.dot(ab, bc) / (numpy.linalg.norm(ab) * numpy.linalg.norm(bc)),
                ),
                unit.radian,
            )

        else:
            return Quantity(
                numpy.arccos(
                    (ac**2 - ab**2 - bc**2) / (-2 * ab * bc),
                ),
                unit.radian,
            )
