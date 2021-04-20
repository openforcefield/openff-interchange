import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Union

from intermol.gromacs import _group_energy_terms
from openff.toolkit.utils.utils import temporary_cd
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import GMXEnergyError, GMXGromppError, GMXMdrunError
from openff.system.tests.energy_tests.report import EnergyReport
from openff.system.utils import get_test_file_path


def _get_mdp_file(key: str) -> Path:
    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
        "cutoff_hbonds": "cutoff_hbonds.mdp",
        "cutoff_buck": "cutoff_buck.mdp",
    }

    return get_test_file_path(f"mdp/{mapping[key]}")


def get_gromacs_energies(
    off_sys: System,
    mdp: str = "cutoff",
    writer: str = "internal",
    electrostatics=True,
) -> EnergyReport:
    """
    Given an OpenFF System object, return single-point energies as computed by GROMACS.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.system.components.system.System
        An OpenFF System object to compute the single-point energy of
    mdp : str, default="cutoff"
        A string key identifying the GROMACS `.mdp` file to be used. See `_get_mdp_file`.
    writer : str, default="internal"
        A string key identifying the backend to be used to write GROMACS files. The
        default value of `"internal"` results in this package's exporters being used.
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
            off_sys.to_gro("out.gro", writer=writer)
            off_sys.to_top("out.top", writer=writer)
            report = _run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=_get_mdp_file(mdp),
                maxwarn=2,
                electrostatics=electrostatics,
            )
            return report


def _run_gmx_energy(
    top_file: Union[Path, str],
    gro_file: Union[Path, str],
    mdp_file: Union[Path, str],
    maxwarn: int = 1,
    electrostatics=True,
):
    """
    Given GROMACS files, return single-point energies as computed by GROMACS.

    Parameters
    ----------
    top_file : str or pathlib.Path
        The path to a GROMACS topology (`.top`) file.
    gro_file : str or pathlib.Path
        The path to a GROMACS coordinate (`.gro`) file.
    mdp_file : str or pathlib.Path
        The path to a GROMACS molecular dynamics parameters (`.mdp`) file.
    maxwarn : int, default=1
        The number of warnings to allow when `gmx grompp` is called (via the `-maxwarn` flag).
    electrostatics : bool, default=True
        A boolean indicated whether or not electrostatics should be included in the energy
        calculation.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    grompp_cmd = f"gmx grompp --maxwarn {maxwarn} -o out.tpr"
    grompp_cmd += f" -f {mdp_file} -c {gro_file} -p {top_file}"

    grompp = subprocess.Popen(
        grompp_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = grompp.communicate()

    if grompp.returncode:
        raise GMXGromppError(err)

    mdrun_cmd = "gmx mdrun -s out.tpr -e out.edr"

    mdrun = subprocess.Popen(
        mdrun_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = mdrun.communicate()

    if mdrun.returncode:
        raise GMXMdrunError(err)

    energy_cmd = "gmx energy -f out.edr -o out.xvg"
    stdin = " ".join(map(str, range(1, 20))) + " 0 "

    energy = subprocess.Popen(
        energy_cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = energy.communicate(input=stdin)

    if energy.returncode:
        raise GMXEnergyError(err)

    report = _parse_gmx_energy("out.xvg", electrostatics=electrostatics)

    return report


def _get_gmx_energy_vdw(gmx_energies: Dict):
    """Get the total nonbonded energy from a set of GROMACS energies."""
    gmx_vdw = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["LJ (SR)", "Disper. corr.", "Buck.ham (SR)"]:
        try:
            gmx_vdw += gmx_energies[key]
        except KeyError:
            pass

    return gmx_vdw


def _get_gmx_energy_coul(gmx_energies: Dict, electrostatics: bool = True):
    gmx_coul = 0.0 * omm_unit.kilojoule_per_mole
    if not electrostatics:
        return gmx_coul
    for key in ["Coulomb (SR)", "Coul. recip."]:
        try:
            gmx_coul += gmx_energies[key]
        except KeyError:
            pass

    return gmx_coul


def _get_gmx_energy_torsion(gmx_energies: Dict):
    """Canonicalize torsion energies from a set of GROMACS energies."""
    gmx_torsion = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["Torsion", "Ryckaert-Bell."]:
        try:
            gmx_torsion += gmx_energies[key]
        except KeyError:
            pass

    return gmx_torsion


def _parse_gmx_energy(xvg_path: str, electrostatics: bool):
    """Parse an `.xvg` file written by `gmx energy`."""
    energies, _ = _group_energy_terms(xvg_path)

    # TODO: Better way of filling in missing fields
    # GROMACS may not populate all keys
    for required_key in ["Bond", "Angle", "Proper Dih."]:
        if required_key not in energies:
            energies[required_key] = 0.0 * omm_unit.kilojoule_per_mole

    keys_to_drop = [
        "Kinetic En.",
        "Temperature",
        "Pres. DC",
        "Pressure",
        "Vir-XX",
        "Vir-YY",
        "Vir-ZZ",
        "Vir-YX",
        "Vir-XY",
        "Vir-YZ",
        "Vir-XZ",
    ]
    for key in keys_to_drop:
        if key in energies.keys():
            energies.pop(key)

    report = EnergyReport()

    report.update_energies(
        {
            "Bond": energies["Bond"],
            "Angle": energies["Angle"],
            "Torsion": _get_gmx_energy_torsion(energies),
            "vdW": _get_gmx_energy_vdw(energies),
            "Electrostatics": _get_gmx_energy_coul(
                energies, electrostatics=electrostatics
            ),
        }
    )

    return report
