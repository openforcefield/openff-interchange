from typing import Dict, Optional

from openff.units import unit
from pydantic import validator

from openff.system.models import DefaultModel
from openff.system.types import FloatQuantity

kj_mol = unit.kilojoule / unit.mol


class EnergyError(BaseException):
    """
    Base class for energies in reports not matching.
    """


class MissingEnergyError(BaseException):
    """
    Exception for when one report has a value for an energy group but the other does not.
    """


class EnergyReport(DefaultModel):
    """A lightweight class containing single-point energies as computed by energy tests."""

    # TODO: Use FloatQuantity, not float
    energies: Dict[str, Optional[FloatQuantity]] = {
        "Bond": None,
        "Angle": None,
        "Torsion": None,
        "Nonbonded": None,
    }

    @validator("energies")
    def validate_energies(cls, v):
        for key, val in v.items():
            if not isinstance(val, unit.Quantity):
                v[key] = FloatQuantity.validate_type(val)
        return v

    def update_energies(self, new_energies):
        self.energies.update(self.validate_energies(new_energies))

    # TODO: Better way of exposing tolerances
    def compare(self, other: "EnergyReport", custom_tolerances=None):
        """
        Compare this `EnergyReport` to another `EnergyReport`. Energies are grouped into
        four categories (bond, angle, torsion, and nonbonded) with default tolerances for
        each set to 1e-3 kJ/mol.

        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------
        other: EnergyReport
            The other `EnergyReport` to compare energies against
        custom_tolerances: dict of str: `FloatQuantity`, optional
            Custom energy tolerances to use to use in comparisons.

        """
        tolerances: Dict[str, FloatQuantity] = {
            "Bond": 1e-3 * kj_mol,
            "Angle": 1e-3 * kj_mol,
            "Torsion": 1e-3 * kj_mol,
            "Nonbonded": 1e-3 * kj_mol,
        }

        if custom_tolerances is not None:
            tolerances.update(custom_tolerances)

        tolerances = self.validate_energies(tolerances)
        error = dict()

        for key in self.energies:

            if self.energies[key] is None and other.energies[key] is None:
                continue
            if self.energies[key] is None and other.energies[key] is None:
                raise MissingEnergyError

            diff = self.energies[key] - other.energies[key]  # type: ignore[operator]
            if abs(diff) > tolerances[key]:
                raise EnergyError(
                    key, diff, tolerances[key], self.energies[key], other.energies[key]
                )

            error[key] = diff

        return error

    def __str__(self):
        return (
            "Energies:\n\n"
            f"Bond:     \t\t{self.energies['Bond']}\n"
            f"Angle:    \t\t{self.energies['Angle']}\n"
            f"Torsion:  \t\t{self.energies['Torsion']}\n"
            f"Nonbonded:\t\t{self.energies['Nonbonded']}\n"
        )
