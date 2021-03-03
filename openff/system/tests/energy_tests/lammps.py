import subprocess
from typing import List

from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import LAMMPSRunError
from openff.system.tests.energy_tests.report import EnergyReport


def get_lammps_energies(
    off_sys: System,
    writer: str = "internal",
    electrostatics=True,
) -> EnergyReport:

    from openff.system.interop.internal.lammps import to_lammps

    to_lammps(off_sys, "out.lmp")
    _write_lammps_input(off_sys=off_sys, file_name="tmp.in")

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


def _write_lammps_input(off_sys: System, file_name="test.in"):

    with open(file_name, "w") as fo:
        fo.write(
            "units real\n" "atom_style full\n" "\n" "dimension 3\nboundary p p p\n\n"
        )

        vdw_hander = off_sys.handlers["vdW"]
        electrostatics_handler = off_sys.handlers["Electrostatics"]

        # TODO: Ensure units
        vdw_cutoff = vdw_hander.cutoff  # type: ignore[attr-defined]
        # TODO: Handle separate cutoffs
        coul_cutoff = vdw_cutoff

        fo.write(
            f"pair_style lj/cut/coul/cut {vdw_cutoff} {coul_cutoff}\n"
            "pair_modify mix arithmetic\n\n"
        )

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

        fo.write("read_data out.lmp\n\n")

        fo.write(
            "thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe\n\n"
            "run 0\n"
        )
