"""
Custom non-bonded handlers.
"""
import abc
from typing import Literal

from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    _allow_only,
)
from openff.units import unit

from openff.interchange.components.smirnoff import SMIRNOFFvdWHandler

"""
{
    force_field._parameter_handler_classes[tag]
    for tag in force_field.registered_parameter_handlers
}.intersection(
    force_field._plugin_parameter_handler_classes
)
"""


class _CustomNonbondedHandler(ParameterHandler, abc.ABC):
    def check_handler_compatibility(self, other_handler):
        raise NotImplementedError()

    scale14 = ParameterAttribute(default=0.5, converter=float)

    cutoff = ParameterAttribute(
        default=unit.Quantity(9.0, unit.angstroms),
        unit=unit.angstrom,
    )
    switch_width = ParameterAttribute(
        default=unit.Quantity(1.0 * unit.angstroms),
        unit=unit.angstrom,
    )
    method = ParameterAttribute(
        default="cutoff",
        converter=_allow_only(["cutoff", "PME"]),
    )


class LennardJones14Handler(_CustomNonbondedHandler):
    """Custom handler for 14-6 pseudo-Lennard-Jones potential."""

    class LJ14Type(ParameterType):
        """Custom vdW type for 14-6 pseudo-Lennard-Jones potential."""

        _ELEMENT_NAME = "Atom"

        epsilon = ParameterAttribute(unit=unit.kilocalorie / unit.mole)
        sigma = ParameterAttribute(default=None, unit=unit.angstrom)

    _TAGNAME = "LennardJones14"
    _INFOTYPE = LJ14Type

    potential = ParameterAttribute(
        "Lennard-Jones-14-6", converter=_allow_only("Lennard-Jones-14-6")
    )


class SMIRNOFFLennardJones14Handler(SMIRNOFFvdWHandler):
    """Potential handler for 14-6 pseudo-Lennard-Jones potential."""

    type: Literal["LennardJones14"] = "LennardJones14"

    expression: Literal[
        "4*epsilon*((sigma/r)**14-(sigma/r)**6)"
    ] = "4*epsilon*((sigma/r)**14-(sigma/r)**6)"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [LennardJones14Handler]
