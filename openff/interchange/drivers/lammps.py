"""Functions for running energy evluations with LAMMPS."""
import subprocess
from typing import List, Optional

import numpy as np
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import LAMMPSRunError


def get_lammps_energies(
    off_sys: Interchange,
    round_positions: Optional[int] = None,
    writer: str = "internal",
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by LAMMPS.

    .. warning :: This API is experimental and subject to change.

    .. todo :: Split out _running_ LAMMPS into a separate internal function

    Parameters
    ----------
    off_sys : openff.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    round_positions : int, optional
        The number of decimal places, in nanometers, to round positions. This can be useful when
        comparing to i.e. GROMACS energies, in which positions may be rounded.
    writer : str, default="internal"
        A string key identifying the backend to be used to write LAMMPS files. The
        default value of `"internal"` results in this package's exporters being used.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    if round_positions is not None:
        off_sys.positions = np.round(off_sys.positions, round_positions)

    off_sys.to_lammps("out.lmp")
    mdconfig = MDConfig.from_interchange(off_sys)
    mdconfig.write_lammps_input(
        input_file="tmp.in",
    )

    run_cmd = "lmp_serial -i tmp.in"

    proc = subprocess.Popen(
        run_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = proc.communicate()

    if proc.returncode:
        raise LAMMPSRunError(err)

    # thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe
    parsed_energies = unit.kilocalorie_per_mole * _parse_lammps_log("log.lammps")

    report = EnergyReport(
        energies={
            "Bond": parsed_energies[0],
            "Angle": parsed_energies[1],
            "Torsion": parsed_energies[2] + parsed_energies[3],
            "vdW": parsed_energies[5] + parsed_energies[8],
            "Electrostatics": parsed_energies[6] + parsed_energies[7],
        }
    )

    return report


def _parse_lammps_log(file_in: str) -> List[float]:
    """Parse a LAMMPS log file for energy components."""
    tag = False
    with open(file_in) as fi:
        for line in fi.readlines():
            if tag:
                data = [float(val) for val in line.split()]
                tag = False
            if line.startswith("E_bond"):
                tag = True

    return data
