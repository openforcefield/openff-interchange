# Migrating

##
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

or other simulation enginegs:

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
from openff.interchange.components.interchange import Interchange

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

## Foyer

Where previously Foyer is used to generate a ParmEd Structure:

```python
structure = forcefield.apply(topology)
```

similar objects can be used to generate an `Interchange` object. In the future, direct consumption
of `mbuild.Compound`s will be supported, but at present the only supported topology input is an
OpenFF topology.

```python
topology = _OFFBioTop(other=Molecule.from_smiles("CCO").to_topology())

interchange = Interchange.from_foyer(topology=topology, force_field=forcefield)
```

From here, the `Interchange` object can be exported to any supported files or objects.

Below is a minimal but complete example parameterizing an ethanol molecule with a Foyer force field and creating an OpenMM System alongside GROMACS files:


```python
from foyer import Forcefield
import mdtraj as md
from openff.toolkit.topology import Molecule
from openff.interchange.components.mdtraj import _OFFBioTop
from openff.interchange.components.interchange import Interchange

ethanol: Molecule = Molecule.from_smiles("CCO")
ethanol.generate_conformers(n_conformers=1)
topology: Topology = _OFFBioTop(other=ethanol.to_topology())
topology.mdtop = md.Topology.from_openmm(ethanol.to_topology().to_openmm())

oplsaa: Forcefield = Forcefield(name="oplsaa")

interchange = Interchange.from_foyer(topology=topology, force_field=oplsaa)
interchange["vdW"].mixing_rule = "lorentz-berthelot"
interchange.positions = ethanol.conformers[0]

openmm_system = interchange.to_openmm()

interchange.to_gro("out.gro")
interchange.to_top("out.top")
```
