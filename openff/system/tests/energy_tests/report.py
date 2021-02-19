from typing import Dict

from openff.system.types import DefaultModel

# TODO: Use FloatQuantity, not float
#  kj_mol = unit.Unit("kilojoule / mole")


class EnergyError(BaseException):
    """
    Base class for energies in reports not matching.
    """


class MissingEnergyError(BaseException):
    """
    Exception for when one report has a value for an energy group but the other does not.
    """


class EnergyReport(DefaultModel):

    energies: Dict[str, float] = {
        "Bond": None,
        "Angle": None,
        "Torsion": None,
        "Nonbonded": None,
    }

    # TODO: Expose tolerances
    def compare(self, other):

        tolerances: Dict[str, float] = {
            "Bond": 1e-3,
            "Angle": 1e-3,
            "Torsion": 1e-3,
            "Nonbonded": 1e-3,
        }

        for key in self.energies:
            if self.energies[key] is None and other.energies[key] is None:
                continue
            if self.energies[key] is None and other.energies[key] is None:
                raise MissingEnergyError

            diff = self.energies[key] - other.energies[key]
            if abs(diff) > tolerances[key]:
                raise EnergyError(key, diff)
