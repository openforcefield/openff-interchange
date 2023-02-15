"""Custom classes exposed as plugins."""
from typing import List, Literal, Type

from openff.models.types import FloatQuantity
from openff.toolkit import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    VirtualSiteHandler,
    _allow_only,
)
from openff.units import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff._base import T
from openff.interchange.smirnoff._nonbonded import _SMIRNOFFNonbondedCollection


class BuckinghamHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Buckingham interactions."""

    class BuckinghamType(ParameterType):
        """A custom SMIRNOFF type for Buckingham interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c = ParameterAttribute(
            default=None,
            unit=unit.kilojoule_per_mole * unit.nanometer**6,
        )

    _TAGNAME = "Buckingham"
    _INFOTYPE = BuckinghamType

    scale12 = ParameterAttribute(default=0.0, converter=float)
    scale13 = ParameterAttribute(default=0.0, converter=float)
    scale14 = ParameterAttribute(default=0.5, converter=float)
    scale15 = ParameterAttribute(default=1.0, converter=float)

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="cutoff",
        converter=_allow_only(["cutoff", "PME"]),
    )

    combining_rules = ParameterAttribute(
        default="Lorentz-Berthelot",
        converter=_allow_only(["Lorentz-Berthelot"]),
    )


class SMIRNOFFBuckinghamCollection(_SMIRNOFFNonbondedCollection):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["Buckingham"] = "Buckingham"

    expression: str = "a*exp(-b*r)-c/r**6"

    method: str = "cutoff"

    mixing_rule: str = "Buckingham"

    switch_width: FloatQuantity["angstrom"] = unit.Quantity(1.0, unit.angstrom)  # noqa

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [BuckinghamHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attributes."""
        return ["smirks", "id", "a", "b", "c"]

    def store_potentials(self, parameter_handler: BuckinghamHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        self.method = parameter_handler.method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    "a": parameter.a,
                    "b": parameter.b,
                    "c": parameter.c,
                },
            )

            self.potentials[potential_key] = potential

    @classmethod
    def create(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: BuckinghamHandler,
        topology: Topology,
    ) -> T:
        """
        Create a SMIRNOFFvdWCollection from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(
                f"Found parameter handler type {type(parameter_handler)}, which is not "
                f"supported by potential type {type(cls)}",
            )

        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            method=parameter_handler.method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    @classmethod
    def parameter_handler_precedence(cls) -> List[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["vdw", "VirtualSites"]

    def create_virtual_sites(
        self,
        parameter_handler: VirtualSiteHandler,
        topology: Topology,
    ):
        """create() but with virtual sites."""
        raise NotImplementedError()
