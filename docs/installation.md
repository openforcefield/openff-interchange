# Installation

These instructions assume that the `conda` package manager is installed. If you do not have Conda installed, see the [OpenFF installation documentation](openff.docs:install).

## Quick Installation

Install the latest release of the `openff-interchange` package from `conda-forge`:

```shell
conda install -c conda-forge openff-interchange
```

## Optional dependencies

Some libraries or tools are only used for development, testing, or optional features. If portions of the API that require optional dependencies are called while those package(s) are not available, an informative error message should be provided. If one is not provided or is insufficiently informative, please [raise an issue](https://github.com/openforcefield/openff-interchange/issues).

It is assumed that all packages are updated to their latest minor versions. Compatibility with old releases is not guaranteed and likely to not work. For example, compatibility with older versions of the OpenFF Toolkit (i.e. versions 0.8.3 and older) and OpenMM (7.5.1 and older) are not guaranteed. If there are a compelling reasons to add compatibility with old versions of dependencies, please [raise an issue](https://github.com/openforcefield/openff-interchange/issues).

All packages (with the exception of OpenEye toolkits) are understood to be open source and free to use. Some operations within the OpenFF Toolkit can be faster if OpenEye toolkits are available, but free alternatives (using RDKit, AmberTools) are available for all methods. For more see [here](https://open-forcefield-toolkit.readthedocs.io/en/stable/installation.html#optional-dependencies).

All packages used in core functionality are available on `conda-forge` and it assumed that `conda` is used to install them. Most are also available on [PyPI](https://pypi.org) via `pip`; while this method of installation is likely to work, it is not currently tested and no guarantees are made.
