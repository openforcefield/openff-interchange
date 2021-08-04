"""Functions for running energy evluations with Amber."""
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Union

from openff.utilities.utilities import requires_package, temporary_cd
from simtk import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import SanderError
from openff.interchange.utils import get_test_file_path


def get_amber_energies(
    off_sys: Interchange,
    writer: str = "parmed",
    electrostatics=True,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by Amber.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.interchange.components.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    writer : str, default="parmed"
        A string key identifying the backend to be used to write GROMACS files. The
        default value of `"parmed"` results in ParmEd being used as a backend.
    electrostatics : bool, default=True
        A boolean indicating whether or not electrostatics should be included in the energy
        calculation.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            struct = off_sys._to_parmed()
            struct.save("out.inpcrd")
            struct.save("out.prmtop")
            off_sys.to_top("out.top", writer=writer)
            report = _run_sander(
                prmtop_file="out.prmtop",
                inpcrd_file="out.inpcrd",
                electrostatics=electrostatics,
            )
            return report


@requires_package("intermol")
def _run_sander(
    inpcrd_file: Union[Path, str],
    prmtop_file: Union[Path, str],
    electrostatics=True,
):
    """
    Given Amber files, return single-point energies as computed by Amber.

    Parameters
    ----------
    prmtop_file : str or pathlib.Path
        The path to an Amber topology (`.prmtop`) file.
    inpcrd_file : str or pathlib.Path
        The path to an Amber coordinate (`.inpcrd`) file.
    electrostatics : bool, default=True
        A boolean indicated whether or not electrostatics should be included in the energy
        calculation.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    from intermol.amber import _group_energy_terms

    in_file = get_test_file_path("min.in")
    sander_cmd = (
        f"sander -i {in_file} -c {inpcrd_file} -p {prmtop_file} -o out.mdout -O"
    )

    sander = subprocess.Popen(
        sander_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = sander.communicate()

    if sander.returncode:
        raise SanderError

    energies, _ = _group_energy_terms("mdinfo")

    energy_report = EnergyReport(
        energies={
            "Bond": energies["BOND"],
            "Angle": energies["ANGLE"],
            "Torsion": energies["DIHED"],
            "vdW": _get_amber_energy_vdw(energies),
            "Electrostatics": _get_amber_energy_coul(energies),
        }
    )

    return energy_report


def _get_amber_energy_vdw(amber_energies: Dict):
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_vdw = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["VDWAALS", "1-4 VDW"]:
        try:
            amber_vdw += amber_energies[key]
        except KeyError:
            pass

    return amber_vdw


def _get_amber_energy_coul(amber_energies: Dict):
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_coul = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["EEL", "1-4 EEL"]:
        try:
            amber_coul += amber_energies[key]
        except KeyError:
            pass

    return amber_coul
