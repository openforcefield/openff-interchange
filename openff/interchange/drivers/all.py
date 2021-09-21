"""Functions for running energy evluations with all available engines."""
from typing import TYPE_CHECKING, Dict

from openff.interchange.drivers.amber import get_amber_energies
from openff.interchange.drivers.gromacs import get_gromacs_energies
from openff.interchange.drivers.lammps import get_lammps_energies
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.interchange.drivers.report import EnergyReport

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange


def get_all_energies(interchange: "Interchange") -> Dict[str, EnergyReport]:
    """
    Given an Interchange object, return single-point energies as computed by all available engines.
    """
    return {
        "OpenMM": get_openmm_energies(interchange),
        "GROMACS": get_gromacs_energies(interchange),
        "LAMMPS": get_lammps_energies(interchange),
        "Amber": get_amber_energies(interchange),
    }
