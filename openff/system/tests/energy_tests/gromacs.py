import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Union

from intermol.gromacs import _group_energy_terms
from openff.toolkit.utils.utils import temporary_cd
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.exceptions import GMXRunError
from openff.system.tests.energy_tests.report import EnergyReport
from openff.system.utils import get_test_file_path


def get_mdp_file(key: str) -> Path:
    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
        "cutoff_hbonds": "cutoff_hbonds.mdp",
    }

    return get_test_file_path(f"mdp/{mapping[key]}")


def get_gromacs_energies(
    off_sys: System,
    mdp: str = "cutoff",
    writer: str = "internal",
    electrostatics=True,
) -> EnergyReport:
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            off_sys.to_gro("out.gro", writer=writer)
            off_sys.to_top("out.top", writer=writer)
            return run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=get_mdp_file(mdp),
                maxwarn=2,
                electrostatics=electrostatics,
            )


def run_gmx_energy(
    top_file: Union[Path, str],
    gro_file: Union[Path, str],
    mdp_file: Union[Path, str],
    maxwarn: int = 1,
    electrostatics=True,
):

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
        raise GMXRunError(err)

    mdrun_cmd = "gmx mdrun -deffnm out"

    mdrun = subprocess.Popen(
        mdrun_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, err = mdrun.communicate()

    if mdrun.returncode:
        raise GMXRunError(err)

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
        raise GMXRunError(err)

    return _parse_gmx_energy("out.xvg", electrostatics=electrostatics)


def _get_gmx_energy_nonbonded(gmx_energies: Dict):
    """Get the total nonbonded energy from a set of GROMACS energies"""
    gmx_nonbonded = 0.0 * omm_unit.kilojoule_per_mole
    for key in ["LJ (SR)", "Coulomb (SR)", "Coul. recip.", "Disper. corr."]:
        try:
            gmx_nonbonded += gmx_energies[key]
        except KeyError:
            pass

    return gmx_nonbonded


def _parse_gmx_energy(xvg_path, electrostatics=True):
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

    report.energies.update(
        {
            "Bond": energies["Bond"],
            "Angle": energies["Angle"],
            "Torsion": energies["Proper Dih."],
        }
    )

    if "Ryckaert-Bell." in energies:
        report.energies["Torsion"] += energies["Ryckaert-Bell."]

    if electrostatics is True:
        report.energies.update(
            {
                "Nonbonded": _get_gmx_energy_nonbonded(energies),
            }
        )
    elif electrostatics is False:
        report.energies.update(
            {
                "Nonbonded": energies["LJ (SR)"],
            }
        )

    if "Buck.ham (SR)" in energies:
        report.energies["Nonbonded"] += energies["Buck.ham (SR)"]

    return report
