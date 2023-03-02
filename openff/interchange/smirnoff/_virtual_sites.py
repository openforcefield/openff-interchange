from typing import Dict, Literal, Optional

import numpy
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterHandler,
    VirtualSiteHandler,
)
from openff.units import unit
from pydantic import Field

from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import _validated_list_to_array
from openff.interchange.models import PotentialKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    SMIRNOFFvdWCollection,
)


def _compute_lj_sigma(
    sigma: Optional[unit.Quantity],
    rmin_half: Optional[unit.Quantity],
) -> unit.Quantity:
    return sigma if rmin_half is None else (2.0 * rmin_half / (2.0 ** (1.0 / 6.0)))


class SMIRNOFFVirtualSiteCollection(SMIRNOFFCollection):
    """
    A handler which stores the information necessary to construct virtual sites (virtual particles).
    """

    key_map: Dict[VirtualSiteKey, PotentialKey] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects and PotentialKey objects.",
    )  # type: ignore[assignment]

    type: Literal["VirtualSites"] = "VirtualSites"
    expression: Literal[""] = ""
    virtual_site_key_topology_index_map: Dict[VirtualSiteKey, int] = Field(
        dict(),
        description="A mapping between VirtualSiteKey objects (stored analogously to TopologyKey objects"
        "in other handlers) and topology indices describing the associated virtual site",
    )
    exclusion_policy: Literal[
        "none",
        "minimal",
        "parents",
        "local",
        "neighbors",
        "connected",
        "all",
    ] = "parents"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [VirtualSiteHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler."""
        return [
            "type",
            "name",
            "id",
            "match",
            "smirks",
            "sigma",
            "epsilon",
            "rmin_half",
            "charge_increment",
            "distance",
            "outOfPlaneAngle",
            "inPlaneAngle",
        ]

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: Topology,
    ) -> None:
        """Populate self.key_map with key-val pairs of [VirtualSiteKey, PotentialKey]."""
        if self.key_map:
            self.key_map = dict()

        # Initialze the virtual site index to begin after the topoogy's atoms (0-indexed)
        virtual_site_index = topology.n_atoms

        matches_by_parent = parameter_handler._find_matches_by_parent(topology)

        for parent_index, parameters in matches_by_parent.items():
            for parameter, orientations in parameters:
                for orientation in orientations:
                    orientation_indices = orientation.topology_atom_indices

                    virtual_site_key = VirtualSiteKey(
                        parent_atom_index=parent_index,
                        orientation_atom_indices=orientation_indices,
                        type=parameter.type,
                        name=parameter.name,
                        match=parameter.match,
                    )

                    # TODO: Better way of specifying unique parameters
                    potential_key = PotentialKey(
                        id=" ".join(
                            [parameter.smirks, parameter.name, parameter.match],
                        ),
                        associated_handler="VirtualSites",
                    )
                    self.key_map[virtual_site_key] = potential_key
                    self.virtual_site_key_topology_index_map[
                        virtual_site_key
                    ] = virtual_site_index
                    virtual_site_index += 1

    def store_potentials(  # type: ignore[override]
        self,
        parameter_handler: VirtualSiteHandler,
        vdw_collection: SMIRNOFFvdWCollection,
        electrostatics_collection: SMIRNOFFElectrostaticsCollection,
    ) -> None:
        """Store VirtualSite-specific parameter-like data."""
        if self.potentials:
            self.potentials = dict()
        for virtual_site_key, potential_key in self.key_map.items():
            # TODO: This logic assumes no spaces in the SMIRKS pattern, name or `match` attribute
            smirks, _, _ = potential_key.id.split(" ")
            parameter = parameter_handler.parameters[smirks]

            virtual_site_potential = Potential(
                parameters={
                    "distance": parameter.distance,
                },
            )
            for attr in ["outOfPlaneAngle", "inPlaneAngle"]:
                if hasattr(parameter, attr):
                    virtual_site_potential.parameters.update(
                        {attr: getattr(parameter, attr)},
                    )
            self.potentials[potential_key] = virtual_site_potential

            vdw_key = PotentialKey(id=potential_key.id, associated_handler="vdw")
            vdw_potential = Potential(
                parameters={
                    "sigma": _compute_lj_sigma(parameter.sigma, parameter.rmin_half),
                    "epsilon": parameter.epsilon,
                },
            )
            vdw_collection.key_map[virtual_site_key] = vdw_key
            vdw_collection.potentials[vdw_key] = vdw_potential

            electrostatics_key = PotentialKey(
                id=potential_key.id,
                associated_handler="Electrostatics",
            )
            electrostatics_potential = Potential(
                parameters={
                    "charge_increments": _validated_list_to_array(
                        parameter.charge_increment,
                    ),
                },
            )
            electrostatics_collection.key_map[virtual_site_key] = electrostatics_key
            electrostatics_collection.potentials[
                electrostatics_key
            ] = electrostatics_potential

    def _get_local_frame_weights(self, virtual_site_key: "VirtualSiteKey"):
        if virtual_site_key.type == "BondCharge":
            origin_weight = [1.0, 0.0]
            x_direction = [-1.0, 1.0]
            y_direction = [-1.0, 1.0]
        elif virtual_site_key.type == "MonovalentLonePair":
            origin_weight = [1, 0.0, 0.0]
            x_direction = [-1.0, 1.0, 0.0]
            y_direction = [-1.0, 0.0, 1.0]
        elif virtual_site_key.type == "DivalentLonePair":
            origin_weight = [0.0, 1.0, 0.0]
            x_direction = [0.5, -1.0, 0.5]
            y_direction = [1.0, -1.0, 0.0]
        elif virtual_site_key.type == "TrivalentLonePair":
            origin_weight = [0.0, 1.0, 0.0, 0.0]
            x_direction = [1 / 3, -1.0, 1 / 3, 1 / 3]
            y_direction = [1.0, -1.0, 0.0, 0.0]

        return origin_weight, x_direction, y_direction

    def _get_local_frame_position(self, virtual_site_key: "VirtualSiteKey"):
        potential_key = self.key_map[virtual_site_key]
        potential = self.potentials[potential_key]
        if virtual_site_key.type == "BondCharge":
            distance = potential.parameters["distance"]
            local_frame_position = numpy.asarray([-1.0, 0.0, 0.0]) * distance
        elif virtual_site_key.type == "MonovalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["inPlaneAngle"].m_as(unit.radian)
            psi = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)
            factor = numpy.array(
                [
                    numpy.cos(theta) * numpy.cos(psi),
                    numpy.sin(theta) * numpy.cos(psi),
                    numpy.sin(psi),
                ],
            )
            local_frame_position = factor * distance
        elif virtual_site_key.type == "DivalentLonePair":
            distance = potential.parameters["distance"]
            theta = potential.parameters["outOfPlaneAngle"].m_as(unit.radian)
            factor = numpy.asarray([-1.0 * numpy.cos(theta), 0.0, numpy.sin(theta)])
            local_frame_position = factor * distance
        elif virtual_site_key.type == "TrivalentLonePair":
            distance = potential.parameters["distance"]
            local_frame_position = numpy.asarray([-1.0, 0.0, 0.0]) * distance

        return local_frame_position
