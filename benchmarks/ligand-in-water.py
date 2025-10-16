import os.path
import tempfile
import time

import polars
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from tqdm import tqdm

from openff.interchange import __version__

data_frame_file = f"{__version__}/ligand_in_water.csv"

if os.path.exists(data_frame_file):
    exit("CSV file already exists, exiting")

water = Molecule.from_smiles("O")
paracetamol = Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")

water.generate_conformers(n_conformers=1)
paracetamol.generate_conformers(n_conformers=1)

interchange = ForceField("openff-2.3.0-rc1.offxml").create_interchange(
    Topology.from_molecules(
        [paracetamol] + 10000 * [water],
    ),
)

interchange.box = Quantity([4, 4, 4], "nanometer")

times = list()

with tempfile.TemporaryDirectory() as tmpdir:
    for index in tqdm(range(10), desc="Ligand in water benchmark"):
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
