"""Plugins handling virtual sites."""
from typing import get_args

import numpy
from nonbonded_plugins.nonbonded import SMIRNOFFBuckinghamCollection
from openff.toolkit.typing.engines.smirnoff.parameters import (
    IndexedParameterAttribute,
    ParameterAttribute,
    VirtualSiteHandler,
    _VirtualSiteType,
)
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError
from openff.units import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import _validated_list_to_array
from openff.interchange.models import PotentialKey
from openff.interchange.smirnoff._nonbonded import SMIRNOFFElectrostaticsCollection
from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection


class BuckinghamVirtualSiteHandler(VirtualSiteHandler):
    """A handler for virtual sites compatible with the Buckingham (exp-6) functional form."""

    class BuckinghamVirtualSiteType(VirtualSiteHandler.VirtualSiteType):
        """A type for virtual sites compatible with the Buckingham (exp-6) functional form."""

        _ELEMENT_NAME = "BuckinghamVirtualSite"

        name = ParameterAttribute(default="EP", converter=str)
        type = ParameterAttribute(converter=str)

        match = ParameterAttribute(converter=str)

        distance = ParameterAttribute(unit=unit.angstrom)
        outOfPlaneAngle = ParameterAttribute(unit=unit.degree)
        inPlaneAngle = ParameterAttribute(unit=unit.degree)

        _DEFAULT_A = 0.0 * unit.kilojoule_per_mole
        _DEFAULT_B = 0.0 * unit.nanometer**-1
        _DEFAULT_C = 0.0 * unit.kilojoule_per_mole * unit.nanometer**6

        a = ParameterAttribute(_DEFAULT_A, unit=_DEFAULT_A.units)
        b = ParameterAttribute(_DEFAULT_B, unit=_DEFAULT_B.units)
        c = ParameterAttribute(_DEFAULT_C, unit=_DEFAULT_C.units)

        charge_increment = IndexedParameterAttribute(unit=unit.elementary_charge)

    @classmethod
    def _add_default_init_kwargs(cls, kwargs):
        """Override VirtualSiteHandler._add_default_init_kwargs without rmin_half/epsilon logic."""
        type_ = kwargs.get("type", None)

        if type_ is None:
            raise SMIRNOFFSpecError("the `type` keyword is missing")
        if type_ not in get_args(_VirtualSiteType):
            raise SMIRNOFFSpecError(
                f"'{type_}' is not a supported virtual site type",
            )

        if "charge_increment" in kwargs:
            expected_num_charge_increments = cls._expected_num_charge_increments(
                type_,
            )
            num_charge_increments = len(kwargs["charge_increment"])
            if num_charge_increments != expected_num_charge_increments:
                raise SMIRNOFFSpecError(
                    f"'{type_}' virtual sites expect exactly {expected_num_charge_increments} "
                    f"charge increments, but got {kwargs['charge_increment']} "
                    f"(length {num_charge_increments}) instead.",
                )

        supports_in_plane_angle = cls._supports_in_plane_angle(type_)
        supports_out_of_plane_angle = cls._supports_out_of_plane_angle(type_)

        if not supports_out_of_plane_angle:
            kwargs["outOfPlaneAngle"] = kwargs.get("outOfPlaneAngle", None)
        if not supports_in_plane_angle:
            kwargs["inPlaneAngle"] = kwargs.get("inPlaneAngle", None)

        match = kwargs.get("match", None)

        if match is None:
            raise SMIRNOFFSpecError("the `match` keyword is missing")

        out_of_plane_angle = kwargs.get("outOfPlaneAngle", 0.0 * unit.degree)
        is_in_plane = (
            None
            if not supports_out_of_plane_angle
            else numpy.isclose(out_of_plane_angle.m_as(unit.degree), 0.0)
        )

        if not cls._supports_match(type_, match, is_in_plane):
            raise SMIRNOFFSpecError(
                f"match='{match}' not supported with type='{type_}'" + ""
                if is_in_plane is None
                else f" and is_in_plane={is_in_plane}",
            )

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
            electrostatics_handler.potentials[
                electrostatics_key
            ] = electrostatics_potential
