"""Functions for running energy evluations with GROMACS."""
import pathlib
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Dict, Optional, Union

from openff.units import unit
from openff.utilities.utilities import requires_package, temporary_cd
from pkg_resources import resource_filename

from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import (
    GMXGromppError,
    GMXMdrunError,
    GMXNotFoundError,
)

if TYPE_CHECKING:
    from openff.units.unit import Quantity

    from openff.interchange import Interchange


def _find_gromacs_executable(raise_exception: bool = False) -> Optional[str]:
    """Attempt to locate a GROMACS executable based on commonly-used names."""
    gromacs_executable_names = ["gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d"]

    for name in gromacs_executable_names:
        if which(name):
            return name

    if raise_exception:
        raise GMXNotFoundError
    else:
        return None


def _get_mdp_file(key: str = "auto") -> str:
    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
        "cutoff_hbonds": "cutoff_hbonds.mdp",
        "cutoff_buck": "cutoff_buck.mdp",
    }

    dir_path = resource_filename("openff.interchange", "tests/data/mdp")
    return pathlib.Path(dir_path).joinpath(mapping[key]).as_posix()


def get_gromacs_energies(
    interchange: "Interchange",
    mdp: str = "auto",
    round_positions: int = 8,
    detailed: bool = False,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by GROMACS.

    .. warning :: This API is experimental and subject to change.

    Parameters
    ----------
    interchange : openff.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    mdp : str, default="cutoff"
        A string key identifying the GROMACS `.mdp` file to be used. See `_get_mdp_file`.
    round_positions: int, default=8
        A decimal precision for the positions in the `.gro` file.
    detailed : bool, default=False
        If True, return a detailed report containing the energies of each term.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    return _process(
        _get_gromacs_energies(
            interchange=interchange,
            mdp=mdp,
            round_positions=round_positions,
        ),
        detailed=detailed,
    )


def _get_gromacs_energies(
    interchange: "Interchange",
    mdp: str = "auto",
    round_positions: int = 8,
) -> Dict[str, unit.Quantity]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            interchange.to_gro("out.gro", decimal=round_positions)
            interchange.to_top("out.top")

            if mdp == "auto":
                mdconfig = MDConfig.from_interchange(interchange)
                mdp_file = "tmp.mdp"
                mdconfig.write_mdp_file(mdp_file)
            else:
                mdp_file = _get_mdp_file(mdp)

            return _run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=mdp_file,
                maxwarn=2,
            )


def _run_gmx_energy(
    top_file: Union[Path, str],
    gro_file: Union[Path, str],
    mdp_file: Union[Path, str],
    maxwarn: int = 1,
) -> Dict[str, unit.Quantity]:
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
    energies: Dict[str, unit.Quantity]
        A dictionary of energies, keyed by the GROMACS energy term name.

    """
    gmx = _find_gromacs_executable(raise_exception=True)

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

    return _parse_gmx_energy("out.edr")


def _get_gmx_energy_vdw(gmx_energies: Dict) -> "Quantity":
    """Get the total nonbonded energy from a set of GROMACS energies."""
    gmx_vdw = 0.0 * kj_mol
    for key in ["LJ (SR)", "LJ-14", "Disper. corr.", "Buck.ham (SR)"]:
        if key in gmx_energies:
            gmx_vdw += gmx_energies[key]

    return gmx_vdw


def _get_gmx_energy_coul(gmx_energies: Dict) -> "Quantity":
    gmx_coul = 0.0 * kj_mol
    for key in ["Coulomb (SR)", "Coul. recip.", "Coulomb-14"]:
        if key in gmx_energies:
            gmx_coul += gmx_energies[key]

    return gmx_coul


def _get_gmx_energy_torsion(gmx_energies: Dict) -> "Quantity":
    """Canonicalize torsion energies from a set of GROMACS energies."""
    gmx_torsion = 0.0 * kj_mol

    for key in ["Torsion", "Ryckaert-Bell.", "Proper Dih.", "Per. Imp. Dih."]:
        if key in gmx_energies:
            gmx_torsion += gmx_energies[key]

    return gmx_torsion


@requires_package("panedr")
def _parse_gmx_energy(edr_path: str) -> Dict[str, unit.Quantity]:
    """Parse an `.edr` file written by `gmx energy`."""
    import panedr

    parsed_energies = panedr.edr_to_df("out.edr").to_dict("index")[0.0]
    parsed_energies.pop("Time")

    #   for key in energies:
    #       energies[key] *= kj_mol

    #   # TODO: Better way of filling in missing fields
    #   # GROMACS may not populate all keys
    #   for required_key in ["Bond", "Angle", "Proper Dih."]:
    #       if required_key not in energies:
    #           energies[required_key] = 0.0 * kj_mol

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
        if key in parsed_energies:
            parsed_energies.pop(key)

    return {key: val * kj_mol for key, val in parsed_energies.items()}


def _process(
    energies: Dict[str, unit.Quantity],
    detailed: bool = False,
) -> EnergyReport:
    """Process energies from GROMACS into a standardized format."""
    if detailed:
        return EnergyReport(energies=energies)

    return EnergyReport(
        energies={
            "Bond": energies["Bond"] if "Bond" in energies else 0.0 * kj_mol,
            "Angle": energies["Angle"],
            "Torsion": _get_gmx_energy_torsion(energies),
            "vdW": _get_gmx_energy_vdw(energies),
            "Electrostatics": _get_gmx_energy_coul(energies),
        },
    )
