OpenFF Interchange
==================
[//]: # (Badges)
[![CI Status](https://github.com/openforcefield/openff-interchange/workflows/ci/badge.svg)](https://github.com/openforcefield/openff-interchange/actions?query=branch%3Amaster+workflow%3Aci)  [![Codecov coverage](https://img.shields.io/codecov/c/github/openforcefield/openff-interchange.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/openforcefield/openff-interchange) [![LGTM analysis](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-interchange.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-interchange/context:python)

A project (and object) for storing, manipulating, and converting molecular mechanics data.

**Please note that this software in an early and experimental state and unsuitable for production.**

#### Getting started

The OpenFF Iterchange object serves primarily as a container object for parametrized data. It can currently take in force fields
and chemical topologies via objects in the [OpenFF Toolkit](https://open-forcefield-toolkit.readthedocs.io/) and produce
an object storing parametrized data.

```python3
# Use the OpenFF Toolkit to generate a minimal chemical topology
from openff.toolkit.topology import Molecule, Topology
top = Molecule.from_smiles("C").to_topology()

# Load OpenFF 1.0.0 "Parsley"
from openff.toolkit.typing.engines.smirnoff import ForceField
parsley = ForceField("openff-1.0.0.offxml")

# Create an OpenFF Interchange object
from openff.interchange.components.interchange import Interchange
out = Interchange.from_smirnoff(force_field=parsley, topology=top)

# Define box vectors and assign atomic positions
import numpy as np
out.box = [4, 4, 4] * np.eye(3)
out.positions = np.random.rand(15).reshape((5, 3))  # Repalce with valid data

# Convert the OpenFF Interchnage object to an OpenMM System ...
omm_sys = out.to_openmm(combine_nonbonded_forces=True)

# ... or write to GROMACS files
out.to_gro("out.gro")
out.to_top("out.top")
```

Future releases will include improved support for other file formats such as those used by AMBER and LAMMPS.

Other examples are available via [binder](https://mybinder.org/v2/gh/openforcefield/openff-interchange/master?filepath=%2Fexamples%2F), runnable in a web browser without installing anyting on your computer.

### Copyright

Copyright (c) 2020-2021, Open Force Field Initiative


#### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
