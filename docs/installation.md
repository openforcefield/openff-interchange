# Installation

These instructions are likely in flux. If something looks out of date or is not working, please raise an issue.


## Quick Installation

Install an alpha/beta/RC version with `conda` (or `mamba`) from `conda-forge`:

```shell
conda install openff-interchange -c conda-forge -c conda-forge/label/openff-interchange_rc
```

## Optional dependencies

Some libraries or tools are only used for development, testing, or optional features. This includes:
  - pytest
  - pytest-cov
  - codecov
  - nbval
  - mypy
  - unyt
  - intermol
  - gromacs >=2020
  - panedr
  - lammps
  - mbuild
  - foyer >=0.8.1

If this list looks out of date, please raise an issue.

If portions of the API that require optional dependencies are called while some of those dependencies are not available, an informative error message should be provided. If one is not provided or is insufficiently informative, please raise an issue.

All packages are assumed to be at their latest version. In particular, compatibility with older versions of the OpenFF Toolkit (i.e. versions 0.8.0 and older) are not guaranteed.

All packages (with the exceptions of GROMACS via `bioconda`) are available on `conda-forge` and it assumed that `conda` is used to install them. Most are also available on PyPI via `pip`; while this method of installation is likely to work, it is not currently tested and therefore no guarantee is made.
