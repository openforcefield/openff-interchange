import os.path
import tempfile
import time

import openmm.app
import openmm.unit
import polars
from openff.toolkit import ForceField, Molecule, Topology
from tqdm import tqdm

from openff.interchange import __version__

data_frame_file = f"{__version__}/protein_in_water.csv"

if os.path.exists(data_frame_file):
    exit("CSV file already exists, exiting")

protein_force_field = ForceField("openff_no_water-3.0.0-alpha0.offxml", "tip3p.offxml")

pdb_topology = Topology.from_pdb("ctr148a-2KO1-model-1.pdb")

modeller = openmm.app.Modeller(
    topology=pdb_topology.to_openmm(),
    positions=pdb_topology.get_positions().to_openmm(),
)


modeller.addSolvent(
    forcefield=openmm.app.ForceField("amber99sb.xml", "tip3p.xml"),
    model="tip3p",
    padding=1 * openmm.unit.nanometer,
)

topology = Topology.from_openmm(
    modeller.topology,
    unique_molecules=[*list({*pdb_topology.molecules}), Molecule.from_smiles("O"), Molecule.from_smiles("[Cl-]")],
    positions=modeller.getPositions(),
)

interchange = protein_force_field.create_interchange(topology)

t0 = time.perf_counter()

interchange.to_openmm()

print(f"Time to create OpenMM System: {time.perf_counter() - t0} seconds")


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
