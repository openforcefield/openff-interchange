OpenFF Interchange
==================
[//]: # (Badges)
[![CI Status](https://github.com/openforcefield/openff-interchange/workflows/full_tests/badge.svg)](https://github.com/openforcefield/openff-interchange/actions?query=branch%3Amain+workflow%3Afull_tests)
[![Documentation Status](https://readthedocs.org/projects/openff-interchange/badge/?version=latest)](https://openff-interchange.readthedocs.io/en/latest/?badge=latest)
[![Codecov coverage](https://img.shields.io/codecov/c/github/openforcefield/openff-interchange.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/openforcefield/openff-interchange)
[![LGTM analysis](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-interchange.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-interchange/context:python)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/openforcefield/openff-interchange/main.svg)](https://results.pre-commit.ci/latest/github/openforcefield/openff-interchange/main)

A project (and object) for storing, manipulating, and converting molecular mechanics data.

**Please note that this software in an early and experimental state and unsuitable for production.**

### Installation

```shell
$ conda install openff-interchange -c conda-forge
```

### Getting started

The OpenFF Iterchange object serves primarily as a container object for parametrized data. It can currently take in force fields
and chemical topologies via objects in the [OpenFF Toolkit](https://open-forcefield-toolkit.readthedocs.io/) and produce
an object storing parametrized data.

```python3
from openff.toolkit import Molecule, ForceField
from openff.units import unit
from openff.interchange import Interchange

# Use the OpenFF Toolkit to generate a chemical topology
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
topology = molecule.to_topology()
topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

# Load OpenFF 2.0.0 "Sage"
sage = ForceField("openff-2.0.0.offxml")

# Create an Interchange object
out = Interchange.from_smirnoff(force_field=sage, topology=topology)

# Convert the Interchnage object to an OpenMM System
omm_sys = out.to_openmm(combine_nonbonded_forces=True)

# or write to GROMACS files
out.to_gro("out.gro")
out.to_top("out.top")

# or roundtrip through JSON or other common serialization formats
roundtripped = Interchange.parse_raw(out.json())
```

Future releases will include improved support for other file formats such as those used by AMBER, CHARMM, and LAMMPS.

Other examples are available via [binder](https://mybinder.org/v2/gh/openforcefield/openff-interchange/main?filepath=%2Fexamples%2F), executable in a web browser without installing anyting on your computer.

For more information, please consult the [full documentation](https://openff-interchange.readthedocs.io/).

### Copyright

Copyright (c) 2020-2021, Open Force Field Initiative


#### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
