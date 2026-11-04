#!/usr/bin/env python3
"""Compare benchmark run times between a baseline and a current CSV.

Exits non-zero if any benchmark's time increased by more than --threshold.
"""

import argparse
import pathlib
import sys

import pandas
from openff.utilities.provenance import _get_conda_list_package_versions
from packaging.version import Version

import openff.interchange

RELEASED = Version(_get_conda_list_package_versions()["openff-interchange-base"])

HEAD = Version(openff.interchange.__version__)


def compare(
    file1: pathlib.Path,
    file2: pathlib.Path,
    column_name: str,
    threshold: float,
):
    try:
        released_mean = pandas.read_csv(file1).mean()[column_name]
    except FileNotFoundError:
        sys.exit(f"Released benchmark file not found: {file1}")

    try:
        current_mean = pandas.read_csv(file2).mean()[column_name]
    except FileNotFoundError:
        sys.exit(f"Current benchmark file not found: {file2}")

    print(f"Comparing benchmarks for column '{column_name}' in case '{file1.stem}'")

    if current_mean > released_mean * (1 + threshold):
        raise Exception(
            f"Benchmark regression detected: current mean {current_mean:.2f} exceeds "
            f"released mean {released_mean:.2f} by more than {threshold:.0%}",
        )
    else:
        print(
            f"Benchmark check passed: current mean {current_mean:.2f} is within "
            f"threshold of released mean {released_mean:.2f}, relative difference "
            f"{(current_mean - released_mean) / released_mean:.2%}",
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Max allowed fractional increase (0.20 = 20%%)",
    )
    args = p.parse_args()

    for case, column in [
        ("large_peg", "openmm_time"),
        ("large_peg", "gromacs_time"),
        ("mixed_solvent", "openmm_time"),
        ("mixed_solvent", "gromacs_time"),
        ("ligand_in_water", "openmm_time"),
        ("ligand_in_water", "gromacs_time"),
        ("protein_in_water", "openmm_time"),
        ("protein_in_water", "gromacs_time"),
    ]:
        compare(
            pathlib.Path("benchmarks") / str(RELEASED) / f"{case}.csv",
            pathlib.Path("benchmarks") / str(HEAD) / f"{case}.csv",
            column,
            args.threshold,
        )


if __name__ == "__main__":
    main()
