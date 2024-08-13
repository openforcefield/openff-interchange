import math
from typing import Literal

import numpy
from openff.toolkit import Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterHandler,
    VirtualSiteHandler,
)
from pydantic import Field

from openff.interchange._annotations import _DegreeQuantity, _Quantity
from openff.interchange.components._particles import _VirtualSite
from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import (
    _lookup_virtual_site_parameter,
    _validated_list_to_array,
)
from openff.interchange.models import PotentialKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFElectrostaticsCollection,
    SMIRNOFFvdWCollection,
)

_DEGREES_TO_RADIANS = numpy.pi / 180.0


# The use of `type` as a field name conflicts with the built-in `type()` when used with PEP 585
_ListOfHandlerTypes = list[type[ParameterHandler]]


class SMIRNOFFVirtualSiteCollection(SMIRNOFFCollection):
    """
    A handler which stores the information necessary to construct virtual sites (virtual particles).
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
    def allowed_parameter_handlers(cls) -> _ListOfHandlerTypes:
        """Return a list of allowed types of ParameterHandler classes."""
        return [VirtualSiteHandler]

    @classmethod
    def supported_parameters(cls) -> list[str]:
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

    @classmethod
    def nonbonded_parameters(cls) -> list[str]:
        """Return a list of parameter attributes handling vdW interactions."""
        return ["sigma", "epsilon"]

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
                        virtual_site_type=parameter.type,
                    )
                    self.key_map[virtual_site_key] = potential_key
                    self.virtual_site_key_topology_index_map[virtual_site_key] = virtual_site_index
                    virtual_site_index += 1

    def store_potentials(  # type: ignore[override]
        self,
        parameter_handler: VirtualSiteHandler,
        vdw_collection: SMIRNOFFvdWCollection,
        electrostatics_collection: SMIRNOFFElectrostaticsCollection,
    ):
        """
        Populate self.potentials with key-val pairs of [PotentialKey, Potential].
        """
        if self.potentials:
            self.potentials = dict()
        for virtual_site_key, potential_key in self.key_map.items():
            # TODO: This logic assumes no spaces in the SMIRKS pattern, name or `match` attribute
            smirks, name, match = potential_key.id.split(" ")
            parameter = _lookup_virtual_site_parameter(
                parameter_handler=parameter_handler,
                smirks=smirks,
                name=name,
                match=match,
            )

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
                    parameter_name: getattr(parameter, parameter_name)
                    for parameter_name in self.nonbonded_parameters()
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
            electrostatics_collection.potentials[electrostatics_key] = electrostatics_potential


class _BondChargeVirtualSite(_VirtualSite):
    type: Literal["BondCharge"]
    distance: _Quantity
    orientations: tuple[int, ...]

    @property
    def local_frame_weights(self) -> tuple[list[float], ...]:
        origin_weight = [1.0, 0.0]  # first atom is origin
        x_direction = [-1.0, 1.0]
        y_direction = [-1.0, 1.0]

        return origin_weight, x_direction, y_direction

    @property
    def local_frame_positions(self) -> Quantity:
        distance_unit = self.distance.units
        return Quantity(
            [-self.distance.m, 0.0, 0.0],
            distance_unit,
        )

    @property
    def local_frame_coordinates(self) -> Quantity:
        return Quantity(
            numpy.array(
                [self.distance.m_as(unit.nanometer), 180.0, 0.0],
            ),
        )


class _MonovalentLonePairVirtualSite(_VirtualSite):
    type: Literal["MonovalentLonePair"]
    distance: _Quantity
    out_of_plane_angle: _DegreeQuantity
    in_plane_angle: _DegreeQuantity
    orientations: tuple[int, ...]

    @property
    def local_frame_weights(self) -> tuple[list[float], ...]:
        origin_weight = [1.0, 0.0, 0.0]  # first/zeroth atom is origin
        x_direction = [-1.0, 1.0, 0.0]
        y_direction = [-1.0, 0.0, 1.0]

        return origin_weight, x_direction, y_direction

    @property
    def local_frame_positions(self) -> Quantity:
        theta = self.in_plane_angle.m_as(unit.radian)
        phi = self.out_of_plane_angle.m_as(unit.radian)

        distance_unit = self.distance.units

        return Quantity(
            [
                self.distance.m * math.cos(theta) * math.cos(phi),
                self.distance.m * math.sin(theta) * math.cos(phi),
                self.distance.m * math.sin(phi),
            ],
            distance_unit,
        )

    @property
    def local_frame_coordinates(self) -> Quantity:
        return Quantity(
            numpy.array(
                [
                    self.distance.m_as(unit.nanometer),
                    self.in_plane_angle.m_as(unit.degree),
                    self.out_of_plane_angle.m_as(unit.degree),
                ],
            ),
        )


class _DivalentLonePairVirtualSite(_VirtualSite):
    type: Literal["DivalentLonePair"]
    distance: _Quantity
    out_of_plane_angle: _DegreeQuantity
    orientations: tuple[int, ...]

    @property
    def local_frame_weights(self) -> tuple[list[float], ...]:
        origin_weight = [1.0, 0.0, 0.0]  # first atom is origin
        x_direction = [-1.0, 0.5, 0.5]
        y_direction = [-1.0, 1.0, 0.0]

        return origin_weight, x_direction, y_direction

    @property
    def local_frame_positions(self) -> Quantity:
        theta = self.out_of_plane_angle.m_as(unit.radian)

        distance_unit = self.distance.units

        return Quantity(
            [
                -self.distance.m * math.cos(theta),
                0.0,
                self.distance.m * math.sin(theta),
            ],
            distance_unit,
        )

    @property
    def local_frame_coordinates(self) -> Quantity:
        return Quantity(
            numpy.array(
                [
                    self.distance.m_as(unit.nanometer),
                    180.0,
                    self.out_of_plane_angle.m_as(unit.degree),
                ],
            ),
        )


class _TrivalentLonePairVirtualSite(_VirtualSite):
    type: Literal["TrivalentLonePair"]
    distance: _Quantity
    orientations: tuple[int, ...]

    @property
    def local_frame_weights(self) -> tuple[list[float], ...]:
        origin_weight = [1.0, 0.0, 0.0, 0.0]  # first atom is origin
        x_direction = [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        y_direction = [-1.0, 1.0, 0.0, 0.0]  # Not used (?)

        return origin_weight, x_direction, y_direction

    @property
    def local_frame_positions(self) -> Quantity:
        distance_unit = self.distance.units
        return Quantity(
            [-self.distance.m, 0.0, 0.0],
            distance_unit,
        )

    @property
    def local_frame_coordinates(self) -> Quantity:
        return Quantity(
            numpy.array(
                [self.distance.m_as(unit.nanometer), 180.0, 0.0],
            ),
        )


def _create_virtual_site_object(
    virtual_site_key: VirtualSiteKey,
    virtual_site_potential,
    # interchange: "Interchange",
    # non_bonded_force: openmm.NonbondedForce,
) -> _VirtualSite:
    orientations = virtual_site_key.orientation_atom_indices

    if virtual_site_key.type == "BondCharge":
        return _BondChargeVirtualSite(
            type="BondCharge",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "MonovalentLonePair":
        return _MonovalentLonePairVirtualSite(
            type="MonovalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            in_plane_angle=virtual_site_potential.parameters["inPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "DivalentLonePair":
        return _DivalentLonePairVirtualSite(
            type="DivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            out_of_plane_angle=virtual_site_potential.parameters["outOfPlaneAngle"],
            orientations=orientations,
        )
    elif virtual_site_key.type == "TrivalentLonePair":
        return _TrivalentLonePairVirtualSite(
            type="TrivalentLonePair",
            distance=virtual_site_potential.parameters["distance"],
            orientations=orientations,
        )

    else:
        raise NotImplementedError(virtual_site_key.type)


def _build_local_coordinate_frames(
    interchange,
    virtual_site_collection: SMIRNOFFVirtualSiteCollection,
) -> numpy.ndarray:
    """
    Build local coordinate frames.

    Adapted from an implementation in OpenFF Recharge (see `LICENSE-3RD-PARTY`).

    See Also
    --------
    https://github.com/openforcefield/openff-recharge/blob/0.5.0/openff/recharge/charges/vsite.py#L584

    """
    stacked_frames: list[list] = [[], [], [], []]

    for virtual_site_key, potential_key in virtual_site_collection.key_map.items():
        virtual_site = _create_virtual_site_object(
            virtual_site_key=virtual_site_key,
            virtual_site_potential=virtual_site_collection.potentials[potential_key],
        )

        # positions of all "orientation" atoms, not just the single "parent"
        orientation_coordinates = interchange.positions[
            virtual_site_key.orientation_atom_indices,
            :,
        ].m_as(unit.nanometer)

        local_frame_weights = virtual_site.local_frame_weights

        weighted_coordinates = local_frame_weights @ orientation_coordinates

        origin = weighted_coordinates[0, :]

        xy_plane = weighted_coordinates[1:, :]

        xy_plane_norm = xy_plane / numpy.sqrt(
            (xy_plane * xy_plane).sum(-1),
        ).reshape(-1, 1)

        x_hat = xy_plane_norm[0, :]
        z_hat = numpy.cross(x_hat, xy_plane[1, :])
        y_hat = numpy.cross(z_hat, x_hat)

        stacked_frames[0].append(origin.reshape(1, -1))
        stacked_frames[1].append(x_hat.reshape(1, -1))
        stacked_frames[2].append(y_hat.reshape(1, -1))
        stacked_frames[3].append(z_hat.reshape(1, -1))

    local_frames = numpy.stack([numpy.vstack(frames) for frames in stacked_frames])

    return Quantity(local_frames, unit.nanometer)


def _convert_local_coordinates(
    local_frame_coordinates: numpy.ndarray,
    local_coordinate_frames: numpy.ndarray,
) -> numpy.ndarray:
    d = local_frame_coordinates[:, 0].reshape(-1, 1)

    theta = (local_frame_coordinates[:, 1] * _DEGREES_TO_RADIANS).reshape(-1, 1)
    phi = (local_frame_coordinates[:, 2] * _DEGREES_TO_RADIANS).reshape(-1, 1)

    cos_theta = numpy.cos(theta)
    sin_theta = numpy.sin(theta)

    cos_phi = numpy.cos(phi)
    sin_phi = numpy.sin(phi)

    # Here we use cos(phi) in place of sin(phi) and sin(phi) in place of cos(phi)
    # this is because we want phi=0 to represent a 0 degree angle from the x-y plane
    # rather than 0 degrees from the z-axis.
    vsite_positions = local_coordinate_frames[0] + d * (
        cos_theta * cos_phi * local_coordinate_frames[1]
        + sin_theta * cos_phi * local_coordinate_frames[2]
        + sin_phi * local_coordinate_frames[3]
    )

    return vsite_positions


def _generate_positions(
    interchange,
    virtual_site_collection: SMIRNOFFVirtualSiteCollection,
    conformer: _Quantity | None = None,
) -> Quantity:
    # TODO: Capture these objects instead of generating them on-the-fly so many times

    local_frame_coordinates = numpy.vstack(
        [
            _create_virtual_site_object(
                virtual_site_key,
                virtual_site_collection.potentials[potential_key],
            ).local_frame_coordinates
            for virtual_site_key, potential_key in virtual_site_collection.key_map.items()
        ],
    )

    local_coordinate_frames = _build_local_coordinate_frames(
        interchange,
        virtual_site_collection,
    )

    virtual_site_positions = _convert_local_coordinates(
        local_frame_coordinates,
        local_coordinate_frames,
    )

    return Quantity(
        virtual_site_positions,
        unit.nanometer,
    )
