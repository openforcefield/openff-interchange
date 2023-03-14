from typing import Iterable, Literal, Optional

from openff.models.types import FloatQuantity
from openff.toolkit.typing.engines.smirnoff.parameters import GBSAHandler
from openff.units import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.constants import kcal_mol_a2
from openff.interchange.smirnoff._base import SMIRNOFFCollection


class SMIRNOFFGBSACollection(SMIRNOFFCollection):
    """Collection storing GBSA potentials as produced by a SMIRNOFF force field."""

    type: Literal["GBSA"] = "GBSA"
    expression: str = "GBSA-OBC1"

    gb_model: str = "OBC1"

    solvent_dielectric: FloatQuantity["dimensionless"] = 78.5
    solute_dielectric: FloatQuantity["dimensionless"] = 1.0
    sa_model: Optional[str] = "ACE"
    surface_area_penalty: FloatQuantity["kilocalorie_per_mole / angstrom ** 2"] = (
        5.4 * kcal_mol_a2
    )
    solvent_radius: FloatQuantity["angstrom"] = 1.4 * unit.angstrom

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
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        for potential_key in self.key_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    "radius": force_field_parameters.radius,
                    "scale": unit.Quantity(
                        force_field_parameters.scale,
                        unit.dimensionless,
                    ),
                },
            )

            self.potentials[potential_key] = potential
