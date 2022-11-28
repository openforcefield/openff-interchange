"""
Custom non-bonded handlers.
"""
import abc

from openff.toolkit.typing.engines.smirnoff import (
    ParameterAttribute,
    ParameterHandler,
    _allow_only,
)
from openff.units import unit


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


class LennardJones14(_CustomNonbondedHandler):
    """Custom handler for 14-6 pesudo-Lennard-Jones potential."""

    _TAGNAME = "LennardJones14"

    potential = ParameterAttribute(
        "Lennard-Jones-14-6", converter=_allow_only("Lennard-Jones-14-6")
    )
