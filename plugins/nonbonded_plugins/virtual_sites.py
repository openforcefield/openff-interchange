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

        a = ParameterAttribute(_DEFAULT_A, unit=_DEFAULT_A.units)
        b = ParameterAttribute(_DEFAULT_B, unit=_DEFAULT_B.units)
        c = ParameterAttribute(_DEFAULT_C, unit=_DEFAULT_C.units)

    _TAGNAME = "BuckinghamVirtualSites"
    _INFOTYPE = BuckinghamVirtualSiteType


class BuckinghamVirtualSiteCollection(SMIRNOFFVirtualSiteCollection):
    """A collection storing virtual sites compatible with the Buckingham (exp-6) functional form."""

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
        vdw_handler: SMIRNOFFBuckinghamCollection,
        electrostatics_handler: SMIRNOFFElectrostaticsCollection,
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
                    "sigma": parameter.sigma,
                    "epsilon": parameter.epsilon,
                },
            )
            vdw_handler.key_map[virtual_site_key] = vdw_key
            vdw_handler.potentials[vdw_key] = vdw_potential

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
            electrostatics_handler.key_map[virtual_site_key] = electrostatics_key
            electrostatics_handler.potentials[electrostatics_key] = (
                electrostatics_potential
            )
