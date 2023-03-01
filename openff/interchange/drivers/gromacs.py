"""Functions for running energy evluations with GROMACS."""
import pathlib
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Dict, Optional, Union

from openff.utilities.utilities import requires_package, temporary_cd
from pkg_resources import resource_filename

from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import GMXGromppError, GMXMdrunError

if TYPE_CHECKING:
    from openff.units.unit import Quantity

    from openff.interchange import Interchange


def _find_gromacs_executable() -> Optional[str]:
    """Attempt to locate a GROMACS executable based on commonly-used names."""
    gromacs_executable_names = ["gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d"]

    for name in gromacs_executable_names:
        if which(name):
            return name

    return None


def _get_mdp_file(key: str = "auto") -> str:
    #       if key == "auto":
    #           return "auto_generated.mdp"
    #
    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
        "cutoff_hbonds": "cutoff_hbonds.mdp",
        "cutoff_buck": "cutoff_buck.mdp",
    }

    dir_path = resource_filename("openff.interchange", "tests/data/mdp")
    return pathlib.Path(dir_path).joinpath(mapping[key]).as_posix()


def get_gromacs_energies(
    off_sys: "Interchange",
    mdp: str = "auto",
    writer: str = "internal",
    decimal: int = 8,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by GROMACS.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    off_sys : openff.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    mdp : str, default="cutoff"
        A string key identifying the GROMACS `.mdp` file to be used. See `_get_mdp_file`.
    writer : str, default="internal"
        A string key identifying the backend to be used to write GROMACS files. The
        default value of `"internal"` results in this package's exporters being used.
    decimal : int, default=8
        A decimal precision for the positions in the `.gro` file.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            off_sys.to_gro("out.gro", writer=writer, decimal=decimal)
            off_sys.to_top("out.top", writer=writer)
            if mdp == "auto":
                mdconfig = MDConfig.from_interchange(off_sys)
                mdconfig.write_mdp_file("tmp.mdp")
                mdp_file = "tmp.mdp"
            else:
                mdp_file = _get_mdp_file(mdp)
            report = _run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=mdp_file,
                maxwarn=2,
            )
            return report


def _run_gmx_energy(
    top_file: Union[Path, str],
    gro_file: Union[Path, str],
    mdp_file: Union[Path, str],
    maxwarn: int = 1,
) -> EnergyReport:
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

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    gmx = _find_gromacs_executable()

    grompp_cmd = f"{gmx} grompp --maxwarn {maxwarn} -o out.tpr"
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

    # Some GROMACS builds will want `-ntmpi` instead of `ntomp`
    mdrun_cmd = f"{gmx} mdrun -s out.tpr -e out.edr -ntomp 1"

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

    report = _parse_gmx_energy("out.edr")

    return report


def _get_gmx_energy_vdw(gmx_energies: Dict) -> "Quantity":
    """Get the total nonbonded energy from a set of GROMACS energies."""
    gmx_vdw = 0.0 * kj_mol
    for key in ["LJ (SR)", "LJ-14", "Disper. corr.", "Buck.ham (SR)"]:
        try:
            gmx_vdw += gmx_energies[key]
        except KeyError:
            pass

    return gmx_vdw


def _get_gmx_energy_coul(gmx_energies: Dict) -> "Quantity":
    gmx_coul = 0.0 * kj_mol
    for key in ["Coulomb (SR)", "Coul. recip.", "Coulomb-14"]:
        try:
            gmx_coul += gmx_energies[key]
        except KeyError:
            pass

    return gmx_coul


def _get_gmx_energy_torsion(gmx_energies: Dict) -> "Quantity":
    """Canonicalize torsion energies from a set of GROMACS energies."""
    gmx_torsion = 0.0 * kj_mol

    for key in ["Torsion", "Ryckaert-Bell.", "Proper Dih.", "Per. Imp. Dih."]:
        try:
            gmx_torsion += gmx_energies[key]
        except KeyError:
            pass

    return gmx_torsion


@requires_package("panedr")
def _parse_gmx_energy(edr_path: str) -> EnergyReport:
    """Parse an `.edr` file written by `gmx energy`."""
    import panedr

    if TYPE_CHECKING:
        from pandas import DataFrame

    df: DataFrame = panedr.edr_to_df("out.edr")
    energies_dict: Dict = df.to_dict("index")
    energies = energies_dict[0.0]
    energies.pop("Time")

    for key in energies:
        energies[key] *= kj_mol

    # TODO: Better way of filling in missing fields
    # GROMACS may not populate all keys
    for required_key in ["Bond", "Angle", "Proper Dih."]:
        if required_key not in energies:
            energies[required_key] = 0.0 * kj_mol

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

    report.update(
        {
            "Bond": energies["Bond"],
            "Angle": energies["Angle"],
            "Torsion": _get_gmx_energy_torsion(energies),
            "vdW": _get_gmx_energy_vdw(energies),
            "Electrostatics": _get_gmx_energy_coul(energies),
        },
    )

    return report
