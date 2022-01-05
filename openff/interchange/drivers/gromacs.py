"""Functions for running energy evluations with GROMACS."""
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from openff.units import unit
from openff.utilities.utilities import requires_package, temporary_cd

from openff.interchange.drivers.report import EnergyReport
from openff.interchange.drivers.utils import _infer_constraints
from openff.interchange.exceptions import (
    GMXGromppError,
    GMXMdrunError,
    UnsupportedExportError,
)
from openff.interchange.utils import get_test_file_path

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange
    from openff.interchange.components.smirnoff import SMIRNOFFvdWHandler


kj_mol = unit.kilojoule / unit.mol

MDP_HEADER = """
nsteps                   = 0
nstenergy                = 1000
continuation             = yes
cutoff-scheme            = verlet

DispCorr                 = Ener
"""
_ = """
pbc                      = xyz
coulombtype              = Cut-off
rcoulomb                 = 0.9
vdwtype                 = cutoff
rvdw                     = 0.9
vdw-modifier             = None
DispCorr                 = No
constraints              = none
"""


def _write_mdp_file(openff_sys: "Interchange") -> None:
    with open("auto_generated.mdp", "w") as mdp_file:
        mdp_file.write(MDP_HEADER)

        if openff_sys.box is not None:
            mdp_file.write("pbc = xyz\n")

        if "Electrostatics" in openff_sys.handlers:
            coul_handler = openff_sys.handlers["Electrostatics"]
            coul_method = coul_handler.method
            coul_cutoff = coul_handler.cutoff.m_as(unit.nanometer)
            coul_cutoff = round(coul_cutoff, 4)
            if coul_method == "cutoff":
                mdp_file.write("coulombtype = Cut-off\n")
                mdp_file.write("coulomb-modifier = None\n")
                mdp_file.write(f"rcoulomb = {coul_cutoff}\n")
            elif coul_method == "pme":
                mdp_file.write("coulombtype = PME\n")
                mdp_file.write(f"rcoulomb = {coul_cutoff}\n")
            elif coul_method == "reactionfield":
                mdp_file.write(f"rcoulomb = {coul_cutoff}\n")
                mdp_file.write(f"rcoulomb = {coul_cutoff}\n")
            else:
                raise UnsupportedExportError(
                    f"Electrostatics method {coul_method} not supported"
                )

        if "vdW" in openff_sys.handlers:
            vdw_handler: "SMIRNOFFvdWHandler" = openff_sys.handlers["vdW"]
            vdw_method = vdw_handler.method.lower().replace("-", "")
            vdw_cutoff = vdw_handler.cutoff.m_as(unit.nanometer)  # type: ignore[attr-defined]
            vdw_cutoff = round(vdw_cutoff, 4)
            if vdw_method == "cutoff":
                mdp_file.write("vdwtype = cutoff\n")
            elif vdw_method == "pme":
                mdp_file.write("vdwtype = PME\n")
            else:
                raise UnsupportedExportError(f"vdW method {vdw_method} not supported")
            mdp_file.write(f"rvdw = {vdw_cutoff}\n")
            if getattr(vdw_handler, "switch_width", None) is not None:
                mdp_file.write("vdw-modifier = Potential-switch\n")
                switch_distance = vdw_handler.cutoff - vdw_handler.switch_width
                switch_distance = switch_distance.m_as(unit.nanometer)  # type: ignore
                mdp_file.write(f"rvdw-switch = {switch_distance}\n")

        constraints = _infer_constraints(openff_sys)
        mdp_file.write(f"constraints = {constraints}\n")


def _get_mdp_file(key: str = "auto") -> str:
    if key == "auto":
        return "auto_generated.mdp"

    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
        "cutoff_hbonds": "cutoff_hbonds.mdp",
        "cutoff_buck": "cutoff_buck.mdp",
    }

    return get_test_file_path(f"mdp/{mapping[key]}")


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
    off_sys : openff.interchange.components.interchange.Interchange
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
                _write_mdp_file(off_sys)
            report = _run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=_get_mdp_file(mdp),
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

    mdrun_cmd = "gmx mdrun -s out.tpr -e out.edr -ntmpi 1"

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


def _get_gmx_energy_vdw(gmx_energies: Dict) -> unit.Quantity:
    """Get the total nonbonded energy from a set of GROMACS energies."""
    gmx_vdw = 0.0 * kj_mol
    for key in ["LJ (SR)", "LJ-14", "Disper. corr.", "Buck.ham (SR)"]:
        try:
            gmx_vdw += gmx_energies[key]
        except KeyError:
            pass

    return gmx_vdw


def _get_gmx_energy_coul(gmx_energies: Dict) -> unit.Quantity:
    gmx_coul = 0.0 * kj_mol
    for key in ["Coulomb (SR)", "Coul. recip.", "Coulomb-14"]:
        try:
            gmx_coul += gmx_energies[key]
        except KeyError:
            pass

    return gmx_coul


def _get_gmx_energy_torsion(gmx_energies: Dict) -> unit.Quantity:
    """Canonicalize torsion energies from a set of GROMACS energies."""
    gmx_torsion = 0.0 * kj_mol
    for key in ["Torsion", "Ryckaert-Bell.", "Proper Dih."]:
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
    energies_dict: Dict = df.to_dict("index")  # type: ignore[assignment]
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

    report.update_energies(
        {
            "Bond": energies["Bond"],
            "Angle": energies["Angle"],
            "Torsion": _get_gmx_energy_torsion(energies),
            "vdW": _get_gmx_energy_vdw(energies),
            "Electrostatics": _get_gmx_energy_coul(energies),
        }
    )

    return report
