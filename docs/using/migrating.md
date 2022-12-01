# Migrating from other tools

Interchange provides special support for migrating from the OpenFF Toolkit.

## Migrating from the OpenFF Toolkit

Where previously the OpenFF Toolkit was used to generate an OpenMM system:

```python
openmm_system = forcefield.create_openmm_system(topology)
```

the same objects can be used to generate an `Interchange` object:

```python
interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
```

From here, the `Interchange` object can be exported to files for OpenMM:

```python
openmm_system = interchange.to_openmm()
```

or other simulation engines:

```python
interchange.to_gro("out.gro")
interchange.to_top("out.top")
interchange.to_lammps("out.lmp")
```

Below is a minimal but complete example parameterizing an ethanol molecule with an OpenFF force field and creating an OpenMM System alongside GROMACS files:

```python
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import get_data_file_path
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange

sdf_file_path = get_data_file_path("molecules/ethanol.sdf")
molecule: Molecule = Molecule.from_file(sdf_file_path)
topology: Topology = molecule.to_topology()

sage = ForceField("openff-2.0.0.offxml")

interchange = Interchange.from_smirnoff(force_field=sage, topology=topology)
interchange.positions = molecule.conformers[0]

openmm_system = interchange.to_openmm()

interchange.to_gro("out.gro")
interchange.to_top("out.top")
```
