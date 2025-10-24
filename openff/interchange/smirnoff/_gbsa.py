from collections.abc import Iterable
from typing import Literal

from openff.toolkit import Quantity, Topology
from openff.toolkit.typing.engines.smirnoff.parameters import GBSAHandler

# TODO: Move build_dimension_type functionality here
from openff.interchange._annotations import (
    AfterValidator,
    Annotated,
    WrapSerializer,
    WrapValidator,
    _DimensionlessQuantity,
    _LengthQuantity,
    _unit_validator_factory,
    quantity_json_serializer,
    quantity_validator,
)
from openff.interchange.components.potentials import Potential
from openff.interchange.constants import kcal_mol_a2
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff._base import SMIRNOFFCollection

_KcalMolA2 = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_unit_validator_factory("kilocalorie_per_mole / angstrom ** 2")),
    WrapSerializer(quantity_json_serializer),
]


class SMIRNOFFGBSACollection(SMIRNOFFCollection):
    """Collection storing GBSA potentials as produced by a SMIRNOFF force field."""

    type: Literal["GBSA"] = "GBSA"
    expression: str = "GBSA-OBC1"

    gb_model: str = "OBC1"

    solvent_dielectric: _DimensionlessQuantity = Quantity(78.5, "dimensionless")
    solute_dielectric: _DimensionlessQuantity = Quantity(1.0, "dimensionless")
    sa_model: str | None = "ACE"
    surface_area_penalty: _KcalMolA2 = Quantity(5.4, kcal_mol_a2)
    solvent_radius: _LengthQuantity = Quantity(1.4, "angstrom")

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [GBSAHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attributes."""
        return ["smirks", "radius", "scale"]

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return [1.0, 1.0]

    @classmethod
    def potential_parameters(cls):
        """Return a list of names of parameters included in each potential in this colletion."""
        return ["radius", "scale"]

    def store_potentials(self, parameter_handler: GBSAHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [PotentialKey, Potential].

        """
        for potential_key in self.key_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    "radius": force_field_parameters.radius,
                    "scale": Quantity(force_field_parameters.scale, "dimensionless"),
                },
            )

            self.potentials[potential_key] = potential

    @classmethod
    def create(
        cls,
        parameter_handler: GBSAHandler,
        topology: "Topology",
    ):
        """Instantiate a `SMIRNOFFGBSACollection` from a parameter handler and a topology."""
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(
                f"Found parameter handler type {type(parameter_handler)}, which is not "
                f"supported by potential type {type(cls)}",
            )

        collection = cls(
            gb_model=parameter_handler.gb_model,
            solvent_dielectric=Quantity(
                parameter_handler.solvent_dielectric,
                "dimensionless",
            ),
            solute_dielectric=Quantity(
                parameter_handler.solute_dielectric,
                "dimensionless",
            ),
            solvent_radius=parameter_handler.solvent_radius,
            sa_model=parameter_handler.sa_model,
            surface_area_penalty=parameter_handler.surface_area_penalty,
        )

        collection.store_matches(parameter_handler=parameter_handler, topology=topology)
        collection.store_potentials(parameter_handler=parameter_handler)

        return collection
