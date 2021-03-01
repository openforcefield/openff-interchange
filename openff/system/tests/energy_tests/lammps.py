import subprocess
from typing import List

import numpy as np
from simtk import unit as omm_unit

from openff.system.exceptions import LAMMPSRunError
from openff.system.tests.energy_tests.report import EnergyReport


def run_lammps_energy() -> EnergyReport:
    run_cmd = "lmp_serial -i default.in"

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
    parsed_energies = omm_unit.kilocalorie_per_mole * _parse_lammps_log("log.lammps")
    nonbonded = np.sum([parsed_energies[i] for i in range(4, 9)])

    report = EnergyReport(
        energies={
            "Bond": parsed_energies[0],
            "Angle": parsed_energies[1],
            "Torsion": parsed_energies[2] + parsed_energies[3],
            "Nonbonded": nonbonded,
        }
    )

    return report


def _parse_lammps_log(file_in) -> List[float]:
    tag = False
    with open(file_in) as fi:
        for line in fi.readlines():
            if tag:
                data = [float(val) for val in line.split()]
            if line.startswith("E_bond"):
                tag = True

    return data
