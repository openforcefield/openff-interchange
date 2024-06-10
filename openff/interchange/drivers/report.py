"""Storing and processing results of energy evaluations."""

import warnings

from openff.toolkit import Quantity
from pydantic import validator

from openff.interchange._annotations import _kJMolQuantity
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


class EnergyReport(_BaseModel):
    """A lightweight class containing single-point energies as computed by energy tests."""

    # TODO: Should the default be None or 0.0 kj_mol?
    energies: dict[str, _kJMolQuantity | None] = {
        "Bond": None,
        "Angle": None,
        "Torsion": None,
        "vdW": None,
        "Electrostatics": None,
    }

    @validator("energies")
    def validate_energies(cls, v: dict) -> dict:
        """Validate the structure of a dict mapping keys to energies."""
        for key, val in v.items():
            if key not in _KNOWN_ENERGY_TERMS:
                raise InvalidEnergyError(f"Energy type {key} not understood.")
            if not isinstance(val, Quantity):
                v[key] = _kJMolQuantity.__call__(str(val))

        return v

    @property
    def total_energy(self):
        """Return the total energy."""
        return self["total"]

    def __getitem__(self, item: str) -> _kJMolQuantity | None:
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
        self.energies.update(self.validate_energies(new_energies))

    def compare(
        self,
        other: "EnergyReport",
        tolerances: dict[str, _kJMolQuantity] | None = None,
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
    ) -> dict[str, _kJMolQuantity]:
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
        energy_differences: dict[str, _kJMolQuantity] = dict()

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
                        energy_differences[key] = self[key] - other[key]
                        energy_differences[key] = self[key] - other[key]

                        nonbondeds_processed = True

                        continue

                else:
                    energy_differences["Nonbonded"] = (
                        self._get_nonbonded_energy() - other._get_nonbonded_energy()
                    )

                    nonbondeds_processed = True

                    continue

        return energy_differences

    def __sub__(self, other: "EnergyReport") -> dict[str, _kJMolQuantity]:
        diff = dict()
        for key in self.energies:
            if key not in other.energies:
                warnings.warn(f"Did not find key {key} in second report", stacklevel=2)
                continue
            diff[key]: _kJMolQuantity = self.energies[key] - other.energies[key]  # type: ignore

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

    def _get_nonbonded_energy(self) -> _kJMolQuantity:
        nonbonded_energy = 0.0 * kj_mol
        for key in ("Nonbonded", "vdW", "Electrostatics"):
            if key in self.energies is not None:
                nonbonded_energy += self.energies[key]

        return nonbonded_energy
