import subprocess
from typing import List

import numpy as np
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import LAMMPSRunError
from openff.system.tests.energy_tests.report import EnergyReport


def get_lammps_energies(
    off_sys: System,
    round_positions=None,
    writer: str = "internal",
    electrostatics=True,
) -> EnergyReport:

    if round_positions is not None:
        off_sys.positions = np.round(off_sys.positions, round_positions)

    off_sys.to_lammps("out.lmp")
    _write_lammps_input(
        off_sys=off_sys,
        file_name="tmp.in",
        electrostatics=electrostatics,
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
    parsed_energies = omm_unit.kilocalorie_per_mole * _parse_lammps_log("log.lammps")

    report = EnergyReport(
        energies={
            "Bond": parsed_energies[0],
            "Angle": parsed_energies[1],
            "Torsion": parsed_energies[2] + parsed_energies[3],
            "Nonbonded": parsed_energies[4],
        }
    )

    return report


def _parse_lammps_log(file_in) -> List[float]:
    tag = False
    with open(file_in) as fi:
        for line in fi.readlines():
            if tag:
                data = [float(val) for val in line.split()]
                tag = False
            if line.startswith("E_bond"):
                tag = True

    return data


def _write_lammps_input(
    off_sys: System,
    file_name="test.in",
    electrostatics=False,
):

    with open(file_name, "w") as fo:
        fo.write(
            "units real\n" "atom_style full\n" "\n" "dimension 3\nboundary p p p\n\n"
        )

        if "Bonds" in off_sys.handlers:
            if len(off_sys["Bonds"].potentials) > 0:
                fo.write("bond_style hybrid harmonic\n")
        if "Angles" in off_sys.handlers:
            if len(off_sys["Angles"].potentials) > 0:
                fo.write("angle_style hybrid harmonic\n")
        if "ProperTorsions" in off_sys.handlers:
            if len(off_sys["ProperTorsions"].potentials) > 0:
                fo.write("dihedral_style hybrid fourier\n")
        if "ImproperTorsions" in off_sys.handlers:
            if len(off_sys["ImproperTorsions"].potentials) > 0:
                fo.write("improper_style hybrid cvff \n")

        vdw_hander = off_sys.handlers["vdW"]
        electrostatics_handler = off_sys.handlers["Electrostatics"]

        # TODO: Ensure units
        vdw_cutoff = vdw_hander.cutoff  # type: ignore[attr-defined]
        # TODO: Handle separate cutoffs
        coul_cutoff = vdw_cutoff

        fo.write(
            "special_bonds lj {} {} {} coul {} {} {}\n\n".format(
                0.0,  # vdw_hander.scale12,
                vdw_hander.scale_13,  # type: ignore[attr-defined]
                vdw_hander.scale_14,  # type: ignore[attr-defined]
                0.0,  # electrostatics_handler.scale12,
                electrostatics_handler.scale_13,  # type: ignore[attr-defined]
                electrostatics_handler.scale_14,  # type: ignore[attr-defined]
            )
        )

        if electrostatics:
            fo.write(f"pair_style lj/cut/coul/cut {vdw_cutoff} {coul_cutoff}\n")
        else:
            fo.write(f"pair_style lj/cut {vdw_cutoff}\n")

        fo.write("pair_modify mix arithmetic\n\n")
        fo.write("read_data out.lmp\n\n")
        fo.write(
            "thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe\n\n"
            "run 0\n"
        )
