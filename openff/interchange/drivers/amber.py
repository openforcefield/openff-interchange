"""Functions for running energy evluations with Amber."""
import subprocess
import tempfile
from distutils.spawn import find_executable
from pathlib import Path
from typing import Dict, Union

from openff.units import unit
from openff.utilities.utilities import temporary_cd
from openmm import unit as openmm_unit

from openff.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import (
    AmberError,
    AmberExecutableNotFoundError,
    SanderError,
)


def get_amber_energies(
    off_sys: Interchange,
    writer: str = "internal",
    electrostatics: bool = True,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by Amber.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.interchange.components.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    writer : str, default="internal"
        A string key identifying the backend to be used to write Amber files.
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
            if writer == "internal":
                off_sys.to_inpcrd("out.inpcrd")
                off_sys.to_prmtop("out.prmtop")
            elif writer == "parmed":
                struct = off_sys._to_parmed()
                struct.save("out.inpcrd")
                struct.save("out.prmtop")
            else:
                raise Exception(f"Unsupported `writer` argument {writer}")

            off_sys.mdconfig.write_sander_input_file("run.in")

            report = _run_sander(
                prmtop_file="out.prmtop",
                inpcrd_file="out.inpcrd",
                input_file="run.in",
                electrostatics=electrostatics,
            )
            return report


def _run_sander(
    inpcrd_file: Union[Path, str],
    prmtop_file: Union[Path, str],
    input_file: Union[Path, str],
    electrostatics: bool = True,
) -> EnergyReport:
    """
    Given Amber files, return single-point energies as computed by Amber.

    Parameters
    ----------
    prmtop_file : str or pathlib.Path
        The path to an Amber topology (`.prmtop`) file.
    inpcrd_file : str or pathlib.Path
        The path to an Amber coordinate (`.inpcrd`) file.
    input_file : str or pathlib.Path
        The path to an Amber/sander input (`.in`) file.
    electrostatics : bool, default=True
        A boolean indicated whether or not electrostatics should be included in the energy
        calculation.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    if not find_executable("sander"):
        raise AmberExecutableNotFoundError(
            "Unable to find the 'sander' executable. Please ensure that "
            "the Amber executables are installed and in your PATH."
        )

    sander_cmd = (
        f"sander -i {input_file} -c {inpcrd_file} -p {prmtop_file} -o out.mdout -O"
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
        raise SanderError(err)

    energies = _group_energy_terms("mdinfo")

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


def _group_energy_terms(mdinfo: str) -> Dict[str, openmm_unit.Quantity]:
    """
    Parse AMBER output file and group the energy terms in a dict.

    This code is partially copied from InterMol, see
    https://github.com/shirtsgroup/InterMol/tree/v0.1/intermol/amber/
    """
    with open(mdinfo) as f:
        all_lines = f.readlines()

    # Find where the energy information starts.
    for i, line in enumerate(all_lines):
        # Seems to hit energy minimization
        if line[0:8] == "   NSTEP":
            startline = i + 2
            break
        # Seems to hit MD "runs"
        elif line[0:6] == " NSTEP":
            startline = i
            break
    else:
        raise AmberError(
            "Unable to detect where energy info starts in AMBER "
            "output file: {}".format(mdinfo)
        )

    # Strange ranges for amber file data.
    ranges = [[1, 24], [26, 49], [51, 77]]

    e_out = dict()
    potential = 0 * unit.kilocalories_per_mole
    for line in all_lines[startline + 1 :]:
        if "=" in line:
            for i in range(3):
                r = ranges[i]
                term = line[r[0] : r[1]]
                if "=" in term:
                    energy_type, energy_value = term.strip().split("=")
                    energy_value = float(energy_value) * unit.kilocalories_per_mole
                    potential += energy_value
                    energy_type = energy_type.rstrip()
                    e_out[energy_type] = energy_value
        else:
            break
    e_out["ENERGY"] = potential

    return e_out


def _get_amber_energy_vdw(amber_energies: Dict) -> openmm_unit.Quantity:
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_vdw = 0.0 * unit.kilojoule_per_mole
    for key in ["VDWAALS", "1-4 VDW", "1-4 NB"]:
        try:
            amber_vdw += amber_energies[key]
        except KeyError:
            pass

    return amber_vdw


def _get_amber_energy_coul(amber_energies: Dict) -> openmm_unit.Quantity:
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_coul = 0.0 * unit.kilojoule_per_mole
    for key in ["EEL", "1-4 EEL"]:
        try:
            amber_coul += amber_energies[key]
        except KeyError:
            pass

    return amber_coul
