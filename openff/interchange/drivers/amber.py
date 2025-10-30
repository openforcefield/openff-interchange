"""Functions for running energy evluations with Amber."""

import subprocess
import tempfile
from pathlib import Path
from shutil import which

from openff.toolkit import Quantity
from openff.utilities.utilities import temporary_cd

from openff.interchange import Interchange
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import (
    AmberError,
    AmberExecutableNotFoundError,
    SanderError,
)


def get_amber_energies(
    interchange: Interchange,
    detailed: bool = False,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by Amber.

    Parameters
    ----------
    interchange
        An OpenFF Interchange object to compute the single-point energy of
    detailed
        If True, return a detailed report containing the energies of each

    Returns
    -------
    report
        An `EnergyReport` object containing the single-point energies.

    """
    return _process(
        _get_amber_energies(
            interchange=interchange,
        ),
        detailed=False,
    )


def _get_amber_energies(
    interchange: Interchange,
) -> dict[str, Quantity]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            # TODO: Use to_amber if implemented
            interchange.to_inpcrd("out.inpcrd")
            interchange.to_prmtop("out.prmtop")

            mdconfig = MDConfig.from_interchange(interchange)
            mdconfig.write_sander_input_file("run.in")

            return _run_sander(
                prmtop_file="out.prmtop",
                inpcrd_file="out.inpcrd",
                input_file="run.in",
            )


def _run_sander(
    inpcrd_file: Path | str,
    prmtop_file: Path | str,
    input_file: Path | str,
) -> dict[str, Quantity]:
    """
    Given Amber files, return single-point energies as computed by Amber.

    Parameters
    ----------
    prmtop_file
        The path to an Amber topology (`.prmtop`) file.
    inpcrd_file
        The path to an Amber coordinate (`.inpcrd`) file.
    input_file
        The path to an Amber/sander input (`.in`) file.

    Returns
    -------
    energies
        A dictionary of energies, keyed by the GROMACS energy term name.

    """
    if not which("sander"):
        raise AmberExecutableNotFoundError(
            "Unable to find the 'sander' executable. Please ensure that "
            "the Amber executables are installed and in your PATH.",
        )

    sander_cmd = f"sander -i {input_file} -c {inpcrd_file} -p {prmtop_file} -o out.mdout -O"

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

    return _parse_amber_energy("mdinfo")


def _parse_amber_energy(mdinfo: str) -> dict[str, Quantity]:
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
            f"Unable to detect where energy info starts in AMBER output file: {mdinfo}",
        )

    # Strange ranges for amber file data.
    ranges = [[1, 24], [26, 49], [51, 77]]

    e_out = dict()
    potential = Quantity(0, "kilocalories_per_mole")
    for line in all_lines[startline + 1 :]:
        if "=" in line:
            for i in range(3):
                r = ranges[i]
                term = line[r[0] : r[1]]
                if "=" in term:
                    energy_type, energy_value = term.strip().split("=")
                    try:
                        energy_value = Quantity(float(energy_value), "kilocalories_per_mole")
                    except ValueError as error:
                        raise AmberError(
                            f"Found bad energy value '{energy_value}' associated with energy type '{energy_type}'",
                        ) from error
                    potential += energy_value
                    energy_type = energy_type.rstrip()
                    e_out[energy_type] = energy_value
        else:
            break
    e_out["ENERGY"] = potential

    return e_out


def _get_amber_energy_vdw(amber_energies: dict) -> Quantity:
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_vdw = Quantity(0, "kilojoule_per_mole")
    for key in ["VDWAALS", "1-4 VDW", "1-4 NB"]:
        if key in amber_energies:
            amber_vdw += amber_energies[key]

    return amber_vdw


def _get_amber_energy_coul(amber_energies: dict) -> Quantity:
    """Get the total nonbonded energy from a set of Amber energies."""
    amber_coul = Quantity(0, "kilojoule_per_mole")
    for key in ["EEL", "1-4 EEL"]:
        if key in amber_energies:
            amber_coul += amber_energies[key]

    return amber_coul


def _process(
    energies: dict[str, Quantity],
    detailed: bool = False,
) -> EnergyReport:
    if detailed:
        return EnergyReport(energies=energies)

    return EnergyReport(
        energies={
            "Bond": energies["BOND"],
            "Angle": energies["ANGLE"],
            "Torsion": energies["DIHED"],
            "vdW": _get_amber_energy_vdw(energies),
            "Electrostatics": _get_amber_energy_coul(energies),
        },
    )
