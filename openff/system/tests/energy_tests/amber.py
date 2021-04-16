import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Union

from intermol.amber import _group_energy_terms
from openff.toolkit.utils.utils import temporary_cd
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import SanderError
from openff.system.tests.energy_tests.report import EnergyReport
from openff.system.utils import get_test_file_path


def get_amber_energies(
    off_sys: System,
    writer: str = "parmed",
    electrostatics=True,
) -> EnergyReport:
    """
    Given an OpenFF System object, return single-point energies as computed by Amber.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.system.components.system.System
        An OpenFF System object to compute the single-point energy of
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
    -------gg
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
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
            "Nonbonded": _get_amber_energy_nonbonded(energies),
        }
    )

    return energy_report


def _get_amber_energy_nonbonded(amber_energies: Dict):
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_nonbonded = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["VDWAALS", "EEL", "1-4 VDW", "1-4 EEL"]:
        try:
            amber_nonbonded += amber_energies[key]
        except KeyError:
            pass

    return amber_nonbonded
