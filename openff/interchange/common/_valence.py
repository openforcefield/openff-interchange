from collections.abc import Iterable
from typing import Literal

from openff.toolkit.topology.molecule import Atom

from openff.interchange.components.potentials import Collection


class BondCollection(Collection):
    """Collection storing bond potentials."""

    type: Literal["Bonds"] = "Bonds"
    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a list of names of parameters included in each potential in this colletion."""
        return "k", "length"

    @classmethod
    def valence_terms(cls, topology) -> list[tuple["Atom", ...]]:
        """Return all bonds in this topology."""
        return [tuple(b.atoms) for b in topology.bonds]


class AngleCollection(Collection):
    """Collection storing Angle potentials."""

    type: Literal["Angles"] = "Angles"
    expression: Literal["k/2*(theta-angle)**2"] = "k/2*(theta-angle)**2"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a list of names of parameters included in each potential in this colletion."""
        return "k", "angle"

    @classmethod
    def valence_terms(cls, topology):
        """Return all angles in this topology."""
        return [angle for angle in topology.angles]


class ProperTorsionCollection(Collection):
    """Handler storing periodic proper torsion potentials."""

    type: Literal["ProperTorsions"] = "ProperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attribute names."""
        return "k", "periodicity", "phase"


class RyckaertBellemansTorsionCollection(Collection):
    """Handler storing Ryckaert-Bellemans torsion potentials."""

    type: Literal["RBTorsions"] = "RBTorsions"
    expression: str = (
        "C0 + C1 * (cos(phi - 180)) "
        "C2 * (cos(phi - 180)) ** 2 + C3 * (cos(phi - 180)) ** 3 "
        "C4 * (cos(phi - 180)) ** 4 + C5 * (cos(phi - 180)) ** 5 "
    )

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attribute names."""
        return "C0", "C1", "C2", "C3", "C4", "C5"


class ImproperTorsionCollection(Collection):
    """Handler storing periodic improper torsion potentials."""

    type: Literal["ImproperTorsions"] = "ImproperTorsions"
    expression: Literal[
        "k*(1+cos(periodicity*theta-phase))"
    ] = "k*(1+cos(periodicity*theta-phase))"

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attribute names."""
        return "k", "periodicity", "phase"
