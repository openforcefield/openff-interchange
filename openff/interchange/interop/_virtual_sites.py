"""
Common helpers for exporting virtual sites.
"""

import abc
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy
from openff.toolkit import Quantity, unit

from openff.interchange.exceptions import (
    MissingPositionsError,
    MissingVirtualSitesError,
)
from openff.interchange.models import BaseVirtualSiteKey
from openff.interchange.pydantic import _BaseModel

if TYPE_CHECKING:
    import openmm

    from openff.interchange import Interchange


class _OpenMMVirtualSite(_BaseModel, abc.ABC):
    particles: list[int]

    @abc.abstractmethod
    def to_openmm(self) -> "openmm.VirtualSite":
        """Create a (subclass of) openmm.VirtualSite"""
        raise NotImplementedError()


class _TwoParticleAverageSite(_OpenMMVirtualSite):
    weights: list[float]

    def to_openmm(self):
        import openmm

        return openmm.TwoParticleAverageSite(
            particle1=self.particles[0],
            particle2=self.particles[1],
            weight1=self.weights[0],
            weight2=self.weights[1],
        )


class _ThreeParticleAverageSite(_OpenMMVirtualSite):
    weights: list[float]

    def to_openmm(self):
        import openmm

        # These arguments can't be named because (vague SWIG reasons)
        return openmm.ThreeParticleAverageSite(
            self.particles[0],  # particle1
            self.particles[1],  # particle2
            self.particles[2],  # particle3
            self.weights[0],  # weight1
            self.weights[1],  # weight2
            self.weights[2],  # weight3
        )


def _virtual_site_parent_molecule_mapping(
    interchange: "Interchange",
) -> dict[BaseVirtualSiteKey, int]:
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
    mapping: dict[BaseVirtualSiteKey, int] = dict()

    if "VirtualSites" not in interchange.collections:
        return mapping

    # TODO: This implicitly assumes the ordering of virtual sites is defined by
    # how they are presented in the iterator; this may cause problems when a
    # molecule (a large ligand? polymer?) has many virtual sites
    for virtual_site_key in interchange["VirtualSites"].key_map:
        assert isinstance(virtual_site_key, BaseVirtualSiteKey), type(virtual_site_key)

        parent_atom_index = virtual_site_key.orientation_atom_indices[0]

        parent_atom = interchange.topology.atom(parent_atom_index)

        parent_molecule = parent_atom.molecule

        mapping[virtual_site_key] = interchange.topology.molecule_index(parent_molecule)

    return mapping


def _get_molecule_virtual_site_map(interchange: "Interchange") -> defaultdict[int, list[BaseVirtualSiteKey]]:
    virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

    molecule_virtual_site_map = defaultdict(list)

    for virtual_site, molecule_index in virtual_site_molecule_map.items():
        molecule_virtual_site_map[molecule_index].append(virtual_site)

    return molecule_virtual_site_map


def get_positions_with_virtual_sites(
    interchange: "Interchange",
    collate: bool = False,
    use_zeros: bool = False,
) -> Quantity:
    """Return the positions of all particles (atoms and virtual sites)."""
    from openff.interchange.interop.common import _build_particle_map

    if interchange.positions is None:
        raise MissingPositionsError(
            f"Positions are required, found {interchange.positions=}.",
        )

    if "VirtualSites" not in interchange.collections:
        raise MissingVirtualSitesError()

    if len(interchange["VirtualSites"].key_map) == 0:
        raise MissingVirtualSitesError()

    # map of molecule index to *list* of virtual site keys contained therein
    molecule_virtual_site_map: defaultdict[int, list[BaseVirtualSiteKey]] = defaultdict(
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

        if collate:
            # for i.e. a 4-site water dimer, these would be

            # {
            #   0: 0,
            #   1: 1,
            #   2: 2,
            #   3: 3,
            #   4: 4,
            #   5: 5,
            #   VirtualSiteKey with atom indices None: 6,
            #   VirtualSiteKey with atom indices None: 7,
            #   }
            uncollated = _build_particle_map(
                interchange=interchange,
                molecule_virtual_site_map=molecule_virtual_site_map,
                collate=False,
            )

            # {
            #   0: 0,
            #   1: 1,
            #   2: 2,
            #   VirtualSiteKey with atom indices None: 3,
            #   3: 4:
            #   4: 5,
            #   5: 6,
            #   VirtualSiteKey with atom indices None: 7,
            #   }
            collated = _build_particle_map(
                interchange=interchange,
                molecule_virtual_site_map=molecule_virtual_site_map,
                collate=True,
            )

            # and so the mapping from collated to uncollated would be
            # [0, 1, 2, 6, 3, 4, 5, 7]
            uncollated_to_collated_mapping = [uncollated[key] for key in collated]

            return numpy.concatenate(
                [
                    interchange.positions,
                    virtual_site_positions,
                ],
            )[uncollated_to_collated_mapping]

        else:
            # could just pass it through a [0, 1, 3, 4, ...] mapping
            # but that seems like a waste

            return numpy.concatenate(
                [
                    interchange.positions,
                    virtual_site_positions,
                ],
            )
    else:
        return interchange.positions


def _get_separation_by_atom_indices(
    interchange: "Interchange",
    atom_indices: tuple[int, ...],
    prioritize_geometry: bool = False,
) -> Quantity:
    """
    Given indices of (two?) atoms, return the distance between them.

    A constraint distance is first searched for, then an equilibrium bond length.

    This is slow, but often necessary for converting virtual site "distances" to weighted
    averages (unitless) of orientation atom positions.
    """
    if prioritize_geometry:
        p1 = interchange.positions[atom_indices[1]]  # type: ignore[index]
        p0 = interchange.positions[atom_indices[0]]  # type: ignore[index]

        return p1 - p0

    if "Constraints" in interchange.collections:
        collection = interchange["Constraints"]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (key.atom_indices[::-1] == atom_indices):
                return collection.potentials[collection.key_map[key]].parameters["distance"]

    if "Bonds" in interchange.collections:
        collection = interchange["Bonds"]  # type: ignore[assignment]

        for key in collection.key_map:
            if (key.atom_indices == atom_indices) or (key.atom_indices[::-1] == atom_indices):
                return collection.potentials[collection.key_map[key]].parameters["length"]

    # Two heavy atoms may be on opposite ends of an angle, in which case it's still
    # possible to determine their separation as defined by the geometry of the force field
    if "Angles" in interchange.collections:
        collection = interchange["Angles"]  # type: ignore[assignment]

        index0 = atom_indices[0]
        index1 = atom_indices[1]
        for key in collection.key_map:
            if (key.atom_indices[0] == index0 and key.atom_indices[2] == index1) or (
                key.atom_indices[2] == index0 and key.atom_indices[0] == index1
            ):
                gamma = collection.potentials[collection.key_map[key]].parameters["angle"]

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

                return Quantity(
                    c2**0.5,
                    unit.nanometer,
                )

    raise ValueError(f"Could not find distance between atoms {atom_indices}")
