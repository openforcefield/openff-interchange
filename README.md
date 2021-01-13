The OpenFF System
=================
[//]: # (Badges)
[![CI Status](https://github.com/openforcefield/openff-system/workflows/ci/badge.svg)](https://github.com/openforcefield/openff-system/actions?query=branch%3Amaster+workflow%3Aci)  [![Codecov coverage](https://img.shields.io/codecov/c/github/openforcefield/openff-system.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/openforcefield/openff-system) [![LGTM analysis](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-system.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-system/context:python)

A molecular system object from the Open Force Field Initiative

**Please note that this software in an early and experimental state and unsuitable for production.**

#### Getting started

The OpenFF System serves primarily as a container object for parametrized data. It can currently take in force fields
and chemical topologies via objects in the [OpenFF Toolkit](https://open-forcefield-toolkit.readthedocs.io/) and produce
an object storing parametrized data.

```python3
# Use the OpenFF Toolkit to generate a minimal chemical topology
from openforcefield.topology import Molecule, Topology
top = Molecule.from_smiles("C").to_topology()

# Load Parsley
from openff.system.stubs import ForceField # Primarily wraps the ForceField class from the OpenFF Toolkit
parsley = ForceField("openff-1.0.0.offxml")

# Create an OpenFF System
off_sys = parsley.create_openff_system(top)

# Define box vectors and assign atomic positions
import numpy as np
off_sys.box = [4, 4, 4] * np.eye(3)
off_sys.positions = np.random.rand(15).reshape((5, 3))  # Repalce with valid data

# Convert the OpenFF System to an OpenMM System
omm_sys = off_sys.to_openmm()
```

Future releases will include support for conversion to other file formats such as those used by AMBER and GROMACS.

Other examples are available via [binder](https://mybinder.org/v2/gh/openforcefield/openff-system/master?filepath=%2Fexamples%2F), runnable in a web browser without installing anyting on your computer.

### Copyright

Copyright (c) 2020, Open Force Field Initiative


#### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
