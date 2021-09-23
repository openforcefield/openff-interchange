"""Functions for running energy evluations with all available engines."""
from typing import TYPE_CHECKING, Dict

from openff.utilities.utilities import requires_package

from openff.interchange.drivers.amber import get_amber_energies
from openff.interchange.drivers.gromacs import get_gromacs_energies
from openff.interchange.drivers.lammps import get_lammps_energies
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.interchange.drivers.report import EnergyReport

if TYPE_CHECKING:
    from pandas import DataFrame

    from openff.interchange.components.interchange import Interchange


def get_all_energies(interchange: "Interchange") -> Dict[str, EnergyReport]:
    """
    Given an Interchange object, return single-point energies as computed by all available engines.
    """
    # TODO: Return something nan-like if one driver fails, but still return others that succeed
    # TODO: Have each driver return the version of the engine that was used

    return {
        "OpenMM": get_openmm_energies(interchange),
        "GROMACS": get_gromacs_energies(interchange),
        "LAMMPS": get_lammps_energies(interchange),
        "Amber": get_amber_energies(interchange),
    }


@requires_package("pandas")
def get_summary_data(interchange: "Interchange") -> "DataFrame":
    """Return a pandas DataFrame with summaries of energies from all available engines."""
    from openff.units import unit
    from pandas import DataFrame

    kj_mol = unit.kilojoule / unit.mol

    energies = get_all_energies(interchange)

    for k, v in energies.items():
        for kk in v.energies:
            energies[k].energies[kk] = energies[k].energies[kk].m_as(kj_mol)

    return DataFrame({k: v.energies for k, v in energies.items()}).T
