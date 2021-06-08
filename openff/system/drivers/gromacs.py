import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from openff.units import unit
from openff.utilities.utilities import requires_package, temporary_cd

from openff.system.drivers.report import EnergyReport
from openff.system.exceptions import (
    GMXGromppError,
    GMXMdrunError,
    UnsupportedExportError,
)
from openff.system.utils import get_test_file_path

if TYPE_CHECKING:
    from openff.system.components.smirnoff import SMIRNOFFvdWHandler
    from openff.system.components.system import System


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


def _write_mdp_file(openff_sys: "System"):
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

        if "Constraints" not in openff_sys.handlers:
            mdp_file.write("constraints = none\n")
        elif "Bonds" not in openff_sys.handlers:
            mdp_file.write("constraints = none\n")
        # TODO: Add support for constraining angles but no bonds?
        else:
            num_constraints = len(openff_sys["Constraints"].slot_map)
            if num_constraints == 0:
                mdp_file.write("constraints = none\n")
            else:
                from openff.system.components.mdtraj import _get_num_h_bonds

                num_h_bonds = _get_num_h_bonds(openff_sys.topology.mdtop)
                num_bonds = len(openff_sys["Bonds"].slot_map)
                num_angles = len(openff_sys["Angles"].slot_map)

                if num_constraints == len(openff_sys["Bonds"].slot_map):
                    mdp_file.write("constraints = all-bonds\n")
                elif num_constraints == num_h_bonds:
                    mdp_file.write("constraints = h-bonds\n")
                elif num_constraints == (num_bonds + num_angles):
                    mdp_file.write("constraints = all-angles\n")


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
    off_sys: "System",
    mdp: str = "auto",
    writer: str = "internal",
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

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            off_sys.to_gro("out.gro", writer=writer)
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


def _get_gmx_energy_vdw(gmx_energies: Dict):
    """Get the total nonbonded energy from a set of GROMACS energies."""
    gmx_vdw = 0.0 * kj_mol
    for key in ["LJ (SR)", "LJ-14", "Disper. corr.", "Buck.ham (SR)"]:
        try:
            gmx_vdw += gmx_energies[key]
        except KeyError:
            pass

    return gmx_vdw


def _get_gmx_energy_coul(gmx_energies: Dict):
    gmx_coul = 0.0 * kj_mol
    for key in ["Coulomb (SR)", "Coul. recip.", "Coulomb-14"]:
        try:
            gmx_coul += gmx_energies[key]
        except KeyError:
            pass

    return gmx_coul


def _get_gmx_energy_torsion(gmx_energies: Dict):
    """Canonicalize torsion energies from a set of GROMACS energies."""
    gmx_torsion = 0.0 * kj_mol
    for key in ["Torsion", "Ryckaert-Bell."]:
        try:
            gmx_torsion += gmx_energies[key]
        except KeyError:
            pass

    return gmx_torsion


@requires_package("panedr")
def _parse_gmx_energy(edr_path: str) -> EnergyReport:
    """Parse an `.xvg` file written by `gmx energy`."""
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
