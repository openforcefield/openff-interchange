import os.path
import tempfile
import time

import numpy
import polars
from openff.toolkit import ForceField, Molecule, Quantity
from tqdm import tqdm

from openff.interchange import __version__

data_frame_file = f"{__version__}/large_peg.csv"

if os.path.exists(data_frame_file):
    exit("CSV file already exists, exiting")


peg = Molecule.from_smiles("CCO" * 500)
peg.add_conformer(
    Quantity(
        numpy.zeros((peg.n_atoms, 3), dtype=float),
        "angstrom",
    ),
)

interchange = ForceField("openff-2.3.0-rc1.offxml").create_interchange(peg.to_topology())
interchange.box = Quantity(numpy.eye(3, dtype=float), "nanometer")

times = list()

with tempfile.TemporaryDirectory() as tmpdir:
    for index in tqdm(range(10), desc="Mixed solvent benchmark"):
        openmm_start = time.perf_counter()
        interchange.to_openmm()
        openmm_end = time.perf_counter()

        gromacs_start = time.perf_counter()
        interchange.to_gromacs(prefix=f"{tmpdir}/temp")
        gromacs_end = time.perf_counter()

        times.append(
            {
                "iteration": index,
                "openmm_time": openmm_end - openmm_start,
                "gromacs_time": gromacs_end - gromacs_start,
            },
        )

data_frame = polars.from_dicts(
    data=times,
    schema={
        "iteration": polars.Int64,
        "openmm_time": polars.Float64,
        "gromacs_time": polars.Float64,
    },
)

data_frame.write_csv(data_frame_file)
