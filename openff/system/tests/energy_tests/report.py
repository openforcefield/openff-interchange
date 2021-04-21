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
        "vdW": None,
        "Electrostatics": None,
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
            "vdW": 1e-3 * kj_mol,
            "Electrostatics": 1e-3 * kj_mol,
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

            # TODO: Remove this when OpenMM's NonbondedForce is split out
            if key == "Nonbonded":
                if "Nonbonded" in other.energies:
                    this_nonbonded = self.energies["Nonbonded"]
                    other_nonbonded = other.energies["Nonbonded"]
                else:
                    this_nonbonded = self.energies["Nonbonded"]
                    other_nonbonded = other.energies["vdW"] + other.energies["Electrostatics"]  # type: ignore
            elif key in ["vdW", "Electrostatics"] and key not in other.energies:
                this_nonbonded = self.energies["vdW"] + self.energies["Electrostatics"]  # type: ignore
                other_nonbonded = other.energies["Nonbonded"]
            else:
                diff = self.energies[key] - other.energies[key]  # type: ignore[operator]
                tolerance = tolerances[key]

                if abs(diff) > tolerance:
                    raise EnergyError(
                        key,
                        diff,
                        tolerance,
                        self.energies[key],
                        other.energies[key],
                    )

                continue

            diff = this_nonbonded - other_nonbonded  # type: ignore
            try:
                tolerance = tolerances[key]
            except KeyError as e:
                if "Nonbonded" in str(e):
                    tolerance = tolerances["vdW"] + tolerances["Electrostatics"]  # type: ignore
                else:
                    raise e

            if abs(diff) > tolerance:
                raise EnergyError(
                    "Nonbonded",
                    diff,
                    tolerance,
                    this_nonbonded,
                    other_nonbonded,
                )

            error[key] = diff

        return error

    def __str__(self):
        return (
            "Energies:\n\n"
            f"Bond:          \t\t{self.energies['Bond']}\n"
            f"Angle:         \t\t{self.energies['Angle']}\n"
            f"Torsion:       \t\t{self.energies['Torsion']}\n"
            f"vdW:           \t\t{self.energies['vdW']}\n"
            f"Electrostatics:\t\t{self.energies['Electrostatics']}\n"
        )
