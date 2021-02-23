import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from intermol.gromacs import _group_energy_terms
from openff.toolkit.utils.utils import temporary_cd

from openff.system.components.system import System
from openff.system.tests.energy_tests.report import EnergyReport
from openff.system.utils import get_test_file_path


def get_mdp_file(key: str) -> Path:
    mapping = {
        "default": "default.mdp",
        "cutoff": "cutoff.mdp",
    }

    return get_test_file_path(f"mdp/{mapping[key]}")


def get_gromacs_energies(
    off_sys: System,
    writer: str = "internal",
) -> EnergyReport:
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            off_sys.to_gro("out.gro", writer=writer)
            off_sys.to_top("out.top", writer=writer)
            gmx_energies = run_gmx_energy(
                top_file="out.top",
                gro_file="out.gro",
                mdp_file=get_mdp_file("default"),
                maxwarn=2,
            )

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
                if key in gmx_energies.keys():
                    gmx_energies.pop(key)

    report = EnergyReport()

    report.energies.update(
        {
            "Bond": gmx_energies["Bond"],
            "Angle": gmx_energies["Angle"],
            "Torsion": gmx_energies["Proper Dih."],
            "Nonbonded": _get_gmx_energy_nonbonded(gmx_energies),
        }
    )

    return report


GMX_PATH = ""


def run_gmx_energy(
    top_file: Path,
    gro_file: Path,
    mdp_file: Path,
    maxwarn: int = 1,
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
        raise Exception

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
        raise Exception

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
        raise Exception

    return _parse_gmx_energy("out.xvg")


def _get_gmx_energy_nonbonded(gmx_energies: Dict):
    """Get the total nonbonded energy from a set of GROMACS energies"""
    gmx_nonbonded = 0.0  # 0.0 * gmx_energies["Potential"].unit
    for key in ["LJ (SR)", "Coulomb (SR)", "Coul. recip.", "Disper. corr."]:
        try:
            gmx_nonbonded += gmx_energies[key]
        except KeyError:
            pass

    return gmx_nonbonded


def _parse_gmx_energy(xvg_path):
    from simtk import unit

    kj_mol = unit.kilojoule_per_mole

    energies, _ = _group_energy_terms(xvg_path)

    # If need to strip units
    for key in energies:
        energies[key] = energies[key] / kj_mol

    # GROMACS may not populate all keys
    for required_key in ["Bond", "Angle", "Proper Dih."]:
        if required_key not in energies:
            energies[required_key] = 0.0

    return energies
