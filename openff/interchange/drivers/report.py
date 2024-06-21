"""Storing and processing results of energy evaluations."""

import warnings
from typing import Annotated

from openff.toolkit import Quantity
from pydantic import BeforeValidator, Field

from openff.interchange._annotations import _Quantity
from openff.interchange.constants import kj_mol
from openff.interchange.exceptions import (
    EnergyError,
    IncompatibleTolerancesError,
    InvalidEnergyError,
)
from openff.interchange.pydantic import _BaseModel

_KNOWN_ENERGY_TERMS: set[str] = {
    "Bond",
    "Angle",
    "Torsion",
    "RBTorsion",
    "Nonbonded",
    "vdW",
    "Electrostatics",
    "vdW 1-4",
    "Electrostatics 1-4",
}


def energies_validator(value: dict[str, Quantity | None]) -> dict[str, Quantity | None]:
    """Validate a dict of energies."""
    if not isinstance(value, dict):
        raise ValueError(f"wrong input type{type(value)}")

    for key, val in value.items():
        if key not in _KNOWN_ENERGY_TERMS:
            raise InvalidEnergyError(f"Energy type {key} not understood.")

        if val is None:
            continue

        if "openmm" in str(type(val)):
            from openff.units.openmm import from_openmm

            value[key] = from_openmm(val).to("kilojoule / mole")
            continue

        if isinstance(val, Quantity):
            value[key] = val.to("kilojoule / mole")

        else:
            raise InvalidEnergyError(f"Energy type {key} not understood.")

    return value


_EnergiesDict = Annotated[
    dict[str, _Quantity | None],
    BeforeValidator(energies_validator),
]


class EnergyReport(_BaseModel):
    """A lightweight class containing single-point energies as computed by energy tests."""

    # TODO: Should the default be None or 0.0 kj_mol?
    energies: _EnergiesDict = Field(
        {
            "Bond": None,
            "Angle": None,
            "Torsion": None,
            "vdW": None,
            "Electrostatics": None,
        },
    )

    @property
    def total_energy(self):
        """Return the total energy."""
        return self["total"]

    def __getitem__(self, item: str) -> Quantity | None:
        if type(item) is not str:
            raise LookupError(
                "Only str arguments can be currently be used for lookups.\n"
                f"Found item {item} of type {type(item)}",
            )
        if item in self.energies.keys():
            return self.energies[item]
        if item.lower() == "total":
            return sum(self.energies.values())  # type: ignore
        else:
            return None

    def update(self, new_energies: dict) -> None:
        """Update the energies in this report with new value(s)."""
        self.energies.update(energies_validator(new_energies))

    def compare(
        self,
        other: "EnergyReport",
        tolerances: dict[str, Quantity] | None = None,
    ):
        """
        Compare two energy reports.

        Parameters
        ----------
        other: EnergyReport
            The other `EnergyReport` to compare energies against

        tolerances: dict of str: Quantity
            Per-key allowed differences in energies

        """
        default_tolerances = {
            "Bond": 1e-3 * kj_mol,
            "Angle": 1e-3 * kj_mol,
            "Torsion": 1e-3 * kj_mol,
            "vdW": 1e-3 * kj_mol,
            "Electrostatics": 1e-3 * kj_mol,
        }

        if tolerances:
            default_tolerances.update(tolerances)

        tolerances = default_tolerances

        # Ensure everything is in kJ/mol for safety of later comparison
        energy_differences = {
            key: diff.to(kj_mol) for key, diff in self.diff(other).items()
        }

        if ("Nonbonded" in tolerances) != ("Nonbonded" in energy_differences):
            raise IncompatibleTolerancesError(
                "Mismatch between energy reports and tolerances with respect to whether nonbonded "
                "interactions are collapsed into a single value.",
            )

        errors = dict()

        for key, diff in energy_differences.items():
            if abs(energy_differences[key]) > tolerances[key]:
                errors[key] = diff

        if errors:
            raise EnergyError(errors)

    def diff(
        self,
        other: "EnergyReport",
    ) -> dict[str, Quantity]:
        """
        Return the per-key energy differences between these reports.

        Parameters
        ----------
        other: EnergyReport
            The other `EnergyReport` to compare energies against

        Returns
        -------
        energy_differences : dict of str: Quantity
            Per-key energy differences

        """
        energy_differences: dict[str, Quantity] = dict()

        nonbondeds_processed = False

        for key in self.energies:
            if key in ("Bond", "Angle", "Torsion"):
                energy_differences[key] = self[key] - other[key]  # type: ignore[operator]

                continue

            if key in ("Nonbonded", "vdW", "Electrostatics"):
                if nonbondeds_processed:
                    continue

                if (self["vdW"] and other["vdW"]) is not None and (
                    self["Electrostatics"] and other["Electrostatics"]
                ) is not None:
                    for key in ("vdW", "Electrostatics"):
                        energy_differences[key] = self[key] - other[key]  # type: ignore[operator]
                        energy_differences[key] = self[key] - other[key]  # type: ignore[operator]

                        nonbondeds_processed = True

                        continue

                else:
                    energy_differences["Nonbonded"] = (
                        self._get_nonbonded_energy() - other._get_nonbonded_energy()
                    )

                    nonbondeds_processed = True

                    continue

        return energy_differences

    def __sub__(self, other: "EnergyReport") -> dict[str, Quantity]:
        diff = dict()
        for key in self.energies:
            if key not in other.energies:
                warnings.warn(f"Did not find key {key} in second report", stacklevel=2)
                continue
            diff[key]: Quantity = self.energies[key] - other.energies[key]  # type: ignore

        return diff

    def __str__(self) -> str:
        return (
            "Energies:\n\n"
            f"Bond:          \t\t{self['Bond']}\n"
            f"Angle:         \t\t{self['Angle']}\n"
            f"Torsion:       \t\t{self['Torsion']}\n"
            f"RBTorsion:     \t\t{self['RBTorsion']}\n"
            f"Nonbonded:     \t\t{self['Nonbonded']}\n"
            f"vdW:           \t\t{self['vdW']}\n"
            f"Electrostatics:\t\t{self['Electrostatics']}\n"
        )

    def _get_nonbonded_energy(self) -> Quantity:
        nonbonded_energy = 0.0 * kj_mol
        for key in ("Nonbonded", "vdW", "Electrostatics"):
            if key in self.energies is not None:
                nonbonded_energy += self.energies[key]

        return nonbonded_energy
