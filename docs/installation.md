# Installation

These instructions are likely in flux. If something looks out of date or is not working, please raise an issue.

## Install dependencies

Until a conda package is live, the core dependencies are listed in `devtools/conda-envs/test_env.yaml` and optional dependencies can be found in `devtools/conda-envs/minimal_env.yaml`.

Required dependencies:
  - python
  - pip
  - numpy
  - pandas
  - pydantic
  - pint
  - openmm
  - openff-toolkit
  - openff-units
  - openff-utilities
  - jax
  - mdtraj
  - ele

Optional/test/dev dependencies:
  - pytest
  - pytest-cov
  - codecov
  - nbval
  - mypy
  - unyt
  - intermol
  - gromacs >=2021
  - panedr
  - lammps
  - mbuild
  - foyer >=0.8.0

If portions of the API that require optional dependencies are called while some of those dependencies are not available, an informative error message should be provided.

All packages are assumed to be at their latest version. In particular, compatibility with older versions of the OpenFF Toolkit (i.e. versions 0.9.0 and older) are not guaranteed.

All packages (with the exceptions of GROMACS via `bioconda`, `openff-units`, and `openff-utilities` described above) are available on `conda-forge` and it assumed that `conda` is used to install them. Most are also available on PyPI via `pip`; while this method of installation is likely to work, it is not currently tested.

## Install the package

Until a conda package is live, the package must be installed locally. After cloning the repo:

```shell
python -m pip install -e .
```
