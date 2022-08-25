"""Storing and processing results of energy evaluations."""
import warnings
from typing import Dict, List, Optional

from openff.units import unit
from pydantic import validator

from openff.interchange.constants import kj_mol
from openff.interchange.exceptions import EnergyError, MissingEnergyError
from openff.interchange.models import DefaultModel
from openff.interchange.types import FloatQuantity


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
    def validate_energies(cls, v: Dict) -> Dict:
        for key, val in v.items():
            if not isinstance(val, unit.Quantity):
                v[key] = FloatQuantity.validate_type(val)
        return v

    @property
    def total_energy(self):
        """Return the total energy."""
        return self["total"]

    def __getitem__(self, item: str) -> Optional[FloatQuantity]:
        if type(item) != str:
            raise LookupError(
                "Only str arguments can be currently be used for lookups.\n"
                f"Found item {item} of type {type(item)}"
            )
        if item in self.energies.keys():
            return self.energies[item]
        if item.lower() == "total":
            return sum(self.energies.values())  # type: ignore
        else:
            return None

    def update_energies(self, new_energies: Dict) -> None:
        """Update the energies in this report with new value(s)."""
        self.energies.update(self.validate_energies(new_energies))

    # TODO: Better way of exposing tolerances
    def compare(
        self,
        other: "EnergyReport",
        custom_tolerances: Optional[Dict[str, FloatQuantity]] = None,
    ) -> None:
        """
        Compare this `EnergyReport` to another `EnergyReport`.

        Energies are grouped into four categories (bond, angle, torsion, and nonbonded) with
        default tolerances for each set to 1e-3 kJ/mol.

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
        errors: List[Dict] = list()

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
                    error: Dict[str, List] = {
                        "key": [key],
                        "diff": [diff],
                        "tol": [tolerance],
                        "ener1": [self.energies[key]],
                        "ener2": [other.energies[key]],
                    }

                    errors.append(error)

                continue

            diff = this_nonbonded - other_nonbonded  # type: ignore
            try:
                tolerance = tolerances[key]
            except KeyError as e:
                if "Nonbonded" in str(e):
                    tolerance = tolerances["vdW"] + tolerances["Electrostatics"]
                else:
                    raise e

            if abs(diff) > tolerance:
                error = {
                    "key": ["Nonbonded"],
                    "diff": [diff],
                    "tol": [tolerance],
                    "ener1": [this_nonbonded],
                    "ener2": [other_nonbonded],
                }
                errors.append(error)

        if len(errors) > 0:
            import pandas

            raise EnergyError(
                "\nSome energy difference(s) exceed tolerances! "
                "\nAll values are reported in kJ/mol:"
                f"\n + {pandas.DataFrame.from_dict(errors).to_string()}"  # type: ignore[arg-type]
            )

        # TODO: Return energy differences even if none are greater than tolerance
        # This might result in mis-matched keys

    def __sub__(self, other: "EnergyReport") -> Dict[str, FloatQuantity]:
        diff = dict()
        for key in self.energies:
            if key not in other.energies:
                warnings.warn(f"Did not find key {key} in second report")
                continue
            diff[key]: FloatQuantity = self.energies[key] - other.energies[key]  # type: ignore

        return diff

    def __str__(self) -> str:
        return (
            "Energies:\n\n"
            f"Bond:          \t\t{self['Bond']}\n"
            f"Angle:         \t\t{self['Angle']}\n"
            f"Torsion:       \t\t{self['Torsion']}\n"
            f"Nonbonded:     \t\t{self['Nonbonded']}\n"
            f"vdW:           \t\t{self['vdW']}\n"
            f"Electrostatics:\t\t{self['Electrostatics']}\n"
        )
