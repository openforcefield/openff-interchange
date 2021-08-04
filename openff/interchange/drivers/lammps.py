"""Functions for running energy evluations with LAMMPS."""
import subprocess
from typing import List

import numpy as np
from openff.units import unit
from simtk import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import LAMMPSRunError


def get_lammps_energies(
    off_sys: Interchange,
    round_positions=None,
    writer: str = "internal",
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by LAMMPS.

    .. warning :: This API is experimental and subject to change.

    .. todo :: Split out _running_ LAMMPS into a separate internal function

    Parameters
    ----------
    off_sys : openff.interchange.components.interchange.Interchange
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
    _write_lammps_input(
        off_sys=off_sys,
        file_name="tmp.in",
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
            "vdW": parsed_energies[5] + parsed_energies[8],
            "Electrostatics": parsed_energies[6] + parsed_energies[7],
        }
    )

    return report


def _parse_lammps_log(file_in) -> List[float]:
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


def _write_lammps_input(
    off_sys: Interchange,
    file_name="test.in",
):
    """Write a LAMMPS input file for running single-point energies."""
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
                fo.write("improper_style cvff\n")

        vdw_hander = off_sys.handlers["vdW"]
        electrostatics_handler = off_sys.handlers["Electrostatics"]

        # TODO: Ensure units
        vdw_cutoff = vdw_hander.cutoff
        vdw_cutoff = vdw_cutoff.m_as(unit.angstrom)

        # TODO: Handle separate cutoffs
        coul_cutoff = vdw_cutoff

        fo.write(
            "special_bonds lj {} {} {} coul {} {} {}\n\n".format(
                0.0,  # vdw_hander.scale12,
                vdw_hander.scale_13,
                vdw_hander.scale_14,
                0.0,  # electrostatics_handler.scale12,
                electrostatics_handler.scale_13,
                electrostatics_handler.scale_14,
            )
        )

        fo.write(f"pair_style lj/cut/coul/cut {vdw_cutoff} {coul_cutoff}\n")

        fo.write("pair_modify mix arithmetic tail yes\n\n")
        fo.write("read_data out.lmp\n\n")
        fo.write(
            "thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe\n\n"
            "run 0\n"
        )
