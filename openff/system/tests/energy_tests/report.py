from typing import Dict, Optional

from simtk import unit as omm_unit

from openff.system.types import DefaultModel


class EnergyError(BaseException):
    """
    Base class for energies in reports not matching.
    """


class MissingEnergyError(BaseException):
    """
    Exception for when one report has a value for an energy group but the other does not.
    """


class EnergyReport(DefaultModel):

    # TODO: Use FloatQuantity, not float
    energies: Dict[str, Optional[omm_unit.Quantity]] = {
        "Bond": None,
        "Angle": None,
        "Torsion": None,
        "Nonbonded": None,
    }

    # TODO: Better way of exposing tolerances
    def compare(self, other, custom_tolerances=None):

        tolerances: Dict[str, float] = {
            "Bond": 1e-3 * omm_unit.kilojoule_per_mole,
            "Angle": 1e-3 * omm_unit.kilojoule_per_mole,
            "Torsion": 1e-3 * omm_unit.kilojoule_per_mole,
            "Nonbonded": 1e-3 * omm_unit.kilojoule_per_mole,
        }

        if custom_tolerances is not None:
            tolerances.update(custom_tolerances)

        for key in self.energies:

            if key == "Nonbonded":
                continue

            if self.energies[key] is None and other.energies[key] is None:
                continue
            if self.energies[key] is None and other.energies[key] is None:
                raise MissingEnergyError

            diff = self.energies[key] - other.energies[key]
            if abs(diff) > tolerances[key]:
                raise EnergyError(key, diff, tolerances[key])
