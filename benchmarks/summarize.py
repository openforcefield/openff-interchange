import subprocess

import packaging.version
import polars

from openff.interchange import __version__

most_recent_release = packaging.version.Version(
    subprocess.check_output(
        ["git", "describe", "--abbrev=0", "--tags"],
    ).decode(),
)


try:
    current_version = f"Current version (commit {__version__.split('+g')[1].split('.')[0]})"
except IndexError:
    current_version = __version__

for header, version in zip(
    ("Most recent release", current_version),
    (most_recent_release, __version__),
):
    print(f"\n=== {header} ({version}) ===")
    systems = {
        "Mixed solvent": f"{version}/mixed_solvent.csv",
        "Ligand in water": f"{version}/ligand_in_water.csv",
        "Protein in water": f"{version}/protein_in_water.csv",
        "500-mer PEG": f"{version}/large_peg.csv",
    }

    for system_name, file_path in systems.items():
        data_frame = polars.read_csv(file_path)
        print(
            f"{system_name} benchmark summary:\n"
            f"\tOpenMM export time (s): "
            f"{data_frame['openmm_time'].mean():.3f} ± {data_frame['openmm_time'].std():.3f}\n"
            f"\tGROMACS export time (s): "
            f"{data_frame['gromacs_time'].mean():.3f} ± {data_frame['gromacs_time'].std():.3f}\n",
        )
