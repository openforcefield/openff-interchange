import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from openff.units import unit
from openff.utilities.utilities import requires_package, temporary_cd
from simtk import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import SanderError, UnsupportedExportError

if TYPE_CHECKING:
    from openff.interchange.components.smirnoff import SMIRNOFFvdWHandler


def _write_in_file(interchange: "Interchange"):
    with open("auto_generated.in", "w") as input_file:
        input_file.write(
            "single-point energy\n" "&cntrl\n" "imin=1,\n" "maxcyc=0,\n" "ntb=1,\n"
        )

        vdw_handler: "SMIRNOFFvdWHandler" = interchange.handlers["vdW"]
        vdw_method = vdw_handler.method.lower().replace("-", "")
        vdw_cutoff = vdw_handler.cutoff.m_as(unit.angstrom)  # type: ignore[attr-defined]
        vdw_cutoff = round(vdw_cutoff, 4)
        if vdw_method == "cutoff":
            input_file.write(f"cut={vdw_cutoff},\n")
        else:
            raise UnsupportedExportError(f"vdW method {vdw_method} not supported")
        if getattr(vdw_handler, "switch_width", None) is not None:
            switch_distance = vdw_handler.cutoff - vdw_handler.switch_width
            switch_distance = switch_distance.m_as(unit.angstrom)  # type: ignore
            switch_distance = round(switch_distance, 4)
            input_file.write(f"fswitch={switch_distance},\n")

        if "Constraints" not in interchange.handlers:
            input_file.write("ntc=2,\n")
        elif "Bonds" not in interchange.handlers:
            input_file.write("ntc=2,\n")
        else:
            num_constraints = len(interchange["Constraints"].slot_map)
            if num_constraints == 0:
                input_file.write("ntc=2,\n")
            else:
                from openff.interchange.components.mdtraj import _get_num_h_bonds

                num_h_bonds = _get_num_h_bonds(interchange.topology.mdtop)
                num_bonds = len(interchange["Bonds"].slot_map)
                num_angles = len(interchange["Angles"].slot_map)

                if num_constraints == len(interchange["Bonds"].slot_map):
                    input_file.write("ntc=3,\n")
                elif num_constraints == num_h_bonds:
                    input_file.write("ntc=3,\n")
                elif num_constraints == (num_bonds + num_angles):
                    raise UnsupportedExportError(
                        "Unclear how to constrain angles with sander"
                    )

        input_file.write("/\n")


def get_amber_energies(
    off_sys: Interchange,
    writer: str = "parmed",
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

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            struct = off_sys._to_parmed()
            struct.save("inpcrd.rst7")
            struct.save("out.prmtop")
            off_sys.to_top("out.top", writer=writer)
            _write_in_file(interchange=off_sys)
            report = _run_sander(
                prmtop_file="out.prmtop",
                inpcrd_file="inpcrd.rst7",
                input_file="auto_generated.in",
            )
            return report


@requires_package("intermol")
def _run_sander(
    inpcrd_file: Union[Path, str],
    prmtop_file: Union[Path, str],
    input_file: Union[Path, str],
):
    """
    Given Amber files, return single-point energies as computed by Amber.

    Parameters
    ----------
    prmtop_file : str or pathlib.Path
        The path to an Amber topology (`.prmtop`) file.
    inpcrd_file : str or pathlib.Path
        The path to an Amber coordinate (`.inpcrd`) file.
    inpcrd_file : str or pathlib.Path
        The path to an Amber input (`.in`) file.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    from intermol.amber import _group_energy_terms

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
