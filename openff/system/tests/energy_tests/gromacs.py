import os
import tempfile

from intermol.gromacs import _group_energy_terms, binaries
from intermol.utils import run_subprocess
from openff.toolkit.utils.utils import temporary_cd
from pkg_resources import resource_filename

from openff.system.components.system import System


def get_gromacs_energies(
    off_sys: System, writer: str = "internal", simple: bool = False
):
    with tempfile.TemporaryDirectory() as tmpdir:
        with temporary_cd(tmpdir):
            off_sys.to_gro("out.gro", writer=writer)
            off_sys.to_top("out.top", writer=writer)
            gmx_energies, energy_file = run_gmx_energy(
                top="out.top",
                gro="out.gro",
                simple=simple,
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
            return gmx_energies, energy_file


GMX_PATH = ""


def run_gmx_energy(
    top, gro, gmx_path=GMX_PATH, grosuff="", simple: bool = False, grompp_check=False
):
    """Compute single-point energies using GROMACS.

    Args:
        top (str):
        gro (str):
        mdp (str):
        grosuff (str):
        grompp_check (bool):

    Returns:
        e_out:
        ener_xvg:

    Note:
        This is copied from the InterMol source code, modified to allow
        for larger values of -maxwarn
    """

    if simple:
        mdp_file = resource_filename("intermol", "tests/gromacs/grompp_vacuum.mdp")
    else:
        mdp_file = resource_filename("intermol", "tests/gromacs/grompp.mdp")

    directory, _ = os.path.split(os.path.abspath(top))

    tpr = os.path.join(directory, "topol.tpr")
    ener = os.path.join(directory, "ener.edr")
    ener_xvg = os.path.join(directory, "energy.xvg")
    conf = os.path.join(directory, "confout.gro")
    mdout = os.path.join(directory, "mdout.mdp")
    state = os.path.join(directory, "state.cpt")
    traj = os.path.join(directory, "traj.trr")
    log = os.path.join(directory, "md.log")
    stdout_path = os.path.join(directory, "gromacs_stdout.txt")
    stderr_path = os.path.join(directory, "gromacs_stderr.txt")

    grompp_bin, mdrun_bin, genergy_bin = binaries(gmx_path, grosuff)

    # Run grompp.
    grompp_bin.extend(["-f", mdp_file, "-c", gro, "-p", top])
    grompp_bin.extend(["-o", tpr, "-po", mdout, "-maxwarn", "2"])
    grompp = run_subprocess(grompp_bin, "gromacs", stdout_path, stderr_path)
    if grompp.returncode != 0:
        raise Exception

    # Run single-point calculation with mdrun.
    mdrun_bin.extend(["-nt", "1", "-s", tpr, "-o", traj])
    mdrun_bin.extend(["-cpo", state, "-c", conf, "-e", ener, "-g", log])
    mdrun = run_subprocess(mdrun_bin, "gromacs", stdout_path, stderr_path)
    if mdrun.returncode != 0:
        raise Exception

    # Extract energies using g_energy
    select = " ".join(map(str, range(1, 20))) + " 0 "
    genergy_bin.extend(["-f", ener, "-o", ener_xvg, "-dp"])
    run_subprocess(genergy_bin, "gromacs", stdout_path, stderr_path, stdin=select)

    return _group_energy_terms(ener_xvg)
