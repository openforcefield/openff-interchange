# OpenFF Interchange

| **Test status** | [![CI Status](https://github.com/openforcefield/openff-interchange/workflows/ci/badge.svg)](https://github.com/openforcefield/openff-interchange/actions?query=branch%3Amain+workflow%3Aci) | [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/openforcefield/openff-interchange/main.svg)](https://results.pre-commit.ci/latest/github/openforcefield/openff-interchange/main) |
| :-- | :-- | :-- |
| **Code quality** | [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) | [![Codecov coverage](https://img.shields.io/codecov/c/github/openforcefield/openff-interchange.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/openforcefield/openff-interchange) |
| **Latest release** | ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/openforcefield/openff-interchange?include_prereleases) | |
| **User support** | [![Documentation Status](https://readthedocs.org/projects/openff-interchange/badge/?version=latest)](https://openff-interchange.readthedocs.io/en/latest/?badge=latest) | [![Discussions](https://img.shields.io/badge/Discussions-GitHub-blue?logo=github)](https://github.com/openforcefield/discussions/discussions) |

A project (and object) for storing, manipulating, and converting molecular mechanics data.

## Documentation

Documentation for Interchange, including examples, a brief user guide, release history, and API docs, is available [on the OpenFF website](https://docs.openforcefield.org/projects/interchange/en/stable). Example notebooks are rendered online among examples from other projects in the [OpenFF ecosysytem docs](https://docs.openforcefield.org/en/latest/examples.html)

## How to cite

Please cite Interchange using the [Zenodo record](https://doi.org/10.5281/zenodo.10068101) of the [latest release](https://zenodo.org/doi/10.5281/zenodo.8147764) or the version that was used. The BibTeX reference of the latest release can be found [at this link](https://zenodo.org/record/8147764/export/hx).

## Installation

Recent versions of the OpenFF Toolkit (0.11.0+, released August 2022) install Interchange by default through its `conda` package.

Interchange can also be installed manually via `conda` (or `mamba`):

```shell
conda install openff-interchange -c conda-forge
```

## Getting started

The `Interchange` object serves primarily as a container object for parametrized data. It can currently take in [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/) or [Foyer](https://foyer.mosdef.org/en/stable/) force fields
and [chemical topologies](https://docs.openforcefield.org/projects/toolkit/en/stable/topology.html) prepared via the [OpenFF Toolkit](https://open-forcefield-toolkit.readthedocs.io/). The resulting object stores parametrized data and provides APIs for export to common formats.

```python3
from openff.toolkit import ForceField, Molecule, Quantity

# Use the OpenFF Toolkit to generate a molecule object from a SMILES pattern
molecule = Molecule.from_smiles("CCO")

# Generate a conformer to be used as atomic coordinates
molecule.generate_conformers(n_conformers=1)

# Convert this molecule to a topology
topology = molecule.to_topology()

# Define periodicity via box vectors
topology.box_vectors = Quantity([4, 4, 4], "nanometer")

# Load OpenFF 2.0.0 "Sage"
sage = ForceField("openff-2.0.0.offxml")

# Create an Interchange object - can also use Interchange.from_smirnoff()
out = sage.create_interchange(force_field=sage, topology=topology)

# Convert the Interchnage object to an OpenMM System
system = out.to_openmm()

# or write to GROMACS files
out.to_gro("out.gro")
out.to_top("out.top")

# or store as JSON
json_blob = out.json()
```

For more information, please consult the [full documentation](https://openff-interchange.readthedocs.io/).

For more examples specific to Interchange, navigate to the `examples/` directory.

## Developing

See the [Development guide](docs/developing.md).

## Copyright

Copyright (c) 2020, Open Force Field Initiative

## License

The source code of Interchange is hosted [on GitHub](https://github.com/openforcefield/openff-interchange) and is available under the MIT license (see the file LICENSE). Some parts inherit from code distributed under other licenses, as detailed in LICENSE-3RD-PARTY.

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
