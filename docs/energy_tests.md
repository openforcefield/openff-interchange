# Energy Tests

This project includes infrastructure for comparing representations of parametrized systems via computing single-point energies using molecular simulation engines.

Currently, OpenMM, GROMACS, and LAMMPS are supported engines for running energy tests.
There are high-level functions for computing energies from constructed `System` objects.
There are also some internal functions that can be used to compute energies from other inputs.

This example shows many ways of computing single-point energies from a constructed `System` object.

```python3
import numpy as np

# Create an OpenFF System object from an OpenFF Molecule, Topology, and ForceField
from openff.toolkit.topology import Molecule
from openff.system.stubs import ForceField

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
topology = molecule.to_topology()
forcefield = ForceField('openff-1.2.0.offxml')
openff_system = forcefield.create_openff_system(topology)
openff_system.box = [5, 5, 5]
openff_system.positions = np.round(molecule.conformers[0]._value / 10.0, 3)


# Directly compute single-point energies via OpenMM, GROMACS, and LAMMPS
from openff.system.tests.energy_tests.openmm import get_openmm_energies
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.lammps import get_lammps_energies

openmm_energies = get_openmm_energies(openff_system)
gmx_energies = get_gromacs_energies(openff_system)
lmp_energies = get_lammps_energies(openff_system)

# Compare energies computed from different engines
openmm_energies.compare(gmx_energies)  # EnergyError: ...
gmx_energies.compare(lmp_energies)  # EnergyError: ...


# Compute single-point energies from engine-specific files/objects
from openff.system.tests.energy_tests.openmm import  _get_openmm_energies

_get_openmm_energies(
    omm_sys=openff_system.to_openmm(),
    box_vectors=openff_system.box,
    positions=openff_system.positions,
)

from openff_system.tests.energy_tests.gromacs import _run_gmx_energy, _get_mdp_file

openff_sys.to_top("out.top", writer="internal")
openff_sys.to_gro("out.gro", writer="internal")

_run_gmx_energy(
    top_file="out.top",
    gro_file="out.gro",
    mdp_file=_get_mdp_file("cutoff_hbonds"),
)
```
