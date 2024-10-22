from typing import Literal

import openmm
from openff.toolkit import Quantity, Topology
from pydantic import Field

from openff.interchange.components.potentials import Collection, Potential
from openff.interchange.models import PotentialKey, VirtualSiteKey


class _VirtualSiteCollection(Collection):
    """
    A handler which stores the information necessary to construct non-SMIRNOFF virtual sites.
    """

    key_map: dict[VirtualSiteKey, PotentialKey] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects and PotentialKey objects.",
    )  # type: ignore[assignment]

    type: Literal["VirtualSites"] = "VirtualSites"
    expression: Literal[""] = ""
    virtual_site_key_topology_index_map: dict[VirtualSiteKey, int] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects (stored analogously to TopologyKey objects"
        "in other handlers) and topology indices describing the associated virtual site",
    )

    exclusion_policy: Literal["parents"] = "parents"


def _convert_virtual_sites(
    system: openmm.System,
    topology: Topology,
) -> _VirtualSiteCollection | None:
    if topology._particle_map is None:
        return None

    collection = _VirtualSiteCollection()

    for particle_index in range(system.getNumParticles()):
        if system.getParticleMass(particle_index)._value != 0.0:
            continue

        virtual_site = system.getVirtualSite(particle_index)

        openmm_particle_indices: tuple[int] = (
            virtual_site.getParticle(0),
            virtual_site.getParticle(1),
            virtual_site.getParticle(2),
        )

        # get ImportedVirtualSiteKey from topology._molecule_virtual_site_map
        for _, list_of_virtual_sites in topology._molecule_virtual_site_map.items():
            for virtual_site_key in list_of_virtual_sites:
                if virtual_site_key.orientation_atom_indices == tuple(
                    topology._particle_map[i] for i in openmm_particle_indices
                ):
                    # TODO: Not the cleanest way to de-duplicate virtual site parameters,
                    #       but other info (i.e. atom name) is insufficient
                    id = type(virtual_site).__name__ + "-".join(
                        map(
                            str,
                            [
                                virtual_site.getWeight(0),
                                virtual_site.getWeight(1),
                                virtual_site.getWeight(2),
                            ],
                        ),
                    )

                    potential_key = PotentialKey(id=id, virtual_site_type="ThreeParticleSite")

                    collection.key_map[virtual_site_key] = potential_key
                    # TODO: This is where geometry-specific information is stored in SMIRNOFF virtual sites,
                    #       but in this case that's not needed
                    collection.potentials[potential_key] = Potential(
                        parameters={
                            "weights": Quantity([virtual_site.getWeight(i) for i in range(3)], "dimensionless"),
                        },
                    )

    return collection
