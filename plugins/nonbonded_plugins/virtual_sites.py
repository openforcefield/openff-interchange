"""Plugins handling virtual sites."""

from nonbonded_plugins.nonbonded import SMIRNOFFBuckinghamCollection
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    VirtualSiteHandler,
    _BaseVirtualSiteType,
)
from openff.units import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import _validated_list_to_array
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.models import PotentialKey
from openff.interchange.smirnoff._nonbonded import SMIRNOFFElectrostaticsCollection
from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection


class BuckinghamVirtualSiteHandler(VirtualSiteHandler):
    """A handler for virtual sites compatible with the Buckingham (exp-6) functional form."""

    class BuckinghamVirtualSiteType(_BaseVirtualSiteType):
        """A type for virtual sites compatible with the Buckingham (exp-6) functional form."""

        _ELEMENT_NAME = "BuckinghamVirtualSite"

        _DEFAULT_A = 0.0 * unit.kilojoule_per_mole
        _DEFAULT_B = 0.0 * unit.nanometer**-1
        _DEFAULT_C = 0.0 * unit.kilojoule_per_mole * unit.nanometer**6

        # `unit` argument must be a Unit object, not a string
        a = ParameterAttribute(default=_DEFAULT_A, unit=_DEFAULT_A.units)
        b = ParameterAttribute(default=_DEFAULT_B, unit=_DEFAULT_B.units)
        c = ParameterAttribute(default=_DEFAULT_C, unit=_DEFAULT_C.units)

    _TAGNAME = "BuckinghamVirtualSites"
    _INFOTYPE = BuckinghamVirtualSiteType


class SMIRNOFFBuckinghamVirtualSiteCollection(SMIRNOFFVirtualSiteCollection):
    """A collection storing virtual sites compatible with the Buckingham (exp-6) functional form."""

    is_plugin: bool = True

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

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [BuckinghamVirtualSiteHandler]

    def store_potentials(  # type: ignore[override]
        self,
        parameter_handler: VirtualSiteHandler,
        vdw_collection: SMIRNOFFBuckinghamCollection,
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
                    parameter_name: getattr(parameter, parameter_name) for parameter_name in self.specific_parameters()
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

    @classmethod
    def create(
        cls,
        parameter_handler,
        topology,
        vdw_collection,
        electrostatics_collection,
    ):
        """
        Create a SMIRNOFFCOllection from toolkit data.
        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(type(parameter_handler))

        collection = cls()

        if hasattr(collection, "fractional_bondorder_method"):
            raise NotImplementedError(
                "Plugins with fractional bond order not yet supported",
            )

        collection.store_matches(parameter_handler=parameter_handler, topology=topology)
        collection.store_potentials(
            parameter_handler=parameter_handler,
            vdw_collection=vdw_collection,
            electrostatics_collection=electrostatics_collection,
        )

        return collection

    @classmethod
    def specific_parameters(cls) -> list[str]:
        """Parameters specific to this collection."""
        return ["a", "b", "c"]
