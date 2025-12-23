# Development guidelines

This document is likely out of date. For the most recent workspace settings, including tasks, environments, and features, see `pyproject.toml` or other configuration files. CI usage is probably defined in `.github/workflows/ci.yaml`.

For topics or details not specified, refer to the [development guidelines]( https://open-forcefield-toolkit.readthedocs.io/en/latest/developing.html) of the OpenFF Toolkit.

## Supported Python versions

Generally, follow [SPEC 0](https://scientific-python.org/specs/spec-0000/). This means that currently Python 3.11-3.13 are supported as of the last update of this document (January 2026). No effort needs to be made to support older versions (Python 2 or 3.10 or earlier) or newer versions that are not well-supported by the [PyData](https://pydata.org) stack.

## Style

Style is enforced with automated linters that run in CI. See `.pre-commit-config.yaml` for the checks that take place, each of which can be run locally with `pre-commit run`.

First, install `pre-commit`

```console
$ mamba install pre-commit -c conda-forge
...
```

or

```console
$ pip install pre-commit
...
```

Then, install the pre-commit hooks (note that it installs the linters into an isolated virtual environment, not the current conda environment):

```console
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

Optionally update the hooks:

```console
$ pre-commit autoupdate
[https://github.com/pre-commit/pre-commit-hooks] already up to date!
[https://github.com/asottile/add-trailing-comma] already up to date!
[https://github.com/astral-sh/ruff-pre-commit] already up to date!
[https://github.com/adamchainz/blacken-docs] already up to date!
[https://github.com/igorshubovych/markdownlint-cli] already up to date!
[https://github.com/kynan/nbstripout] already up to date!
[https://github.com/tox-dev/pyproject-fmt] already up to date!
```

Hooks will now run automatically before commits. Once installed, it should run in a few seconds.

If hooks are installed locally, all linting checks in CI should pass. If hooks are not installed locally or are significatnly out of date, a `pre-commit.ci` bot may commit directly to a PR to make these fixes.

## Type-checking

Type hints are **optional** but encouraged. Invoke with `mypy -p 'openff.interchange'` or `pixi run -e <ENVIRONMENT_NAME> run_mypy`. Check `pyproject.toml` for detailed configuration.

## Documentation

Interchange is documented with Sphinx and hosted by ReadTheDocs at <https://openff-interchange.readthedocs.io>. The documentation is built and served by ReadTheDocs automatically for every pull request --- please check that your changes are documented appropriately!

Interchange uses Autosummary to automatically document the entire public API from the code's docstrings. Docstrings should be written according to the [NumPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html). By default, all modules and members except those beginning with an underscore are included in the API reference. Additional modules such as tests can be excluded from the reference by listing them in the `autosummary_context["exclude_modules"]` variable in `docs/conf.py`.

This is implemented by including a `:recursive:` Autosummary directive in `docs/index.md` and a customised module template `docs/_templates/autosummary/module.rst`. This template produces neatly segmented, complete documentation with a coherent navigation structure.

The remaining parts of the documentation are written in [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/).

### Building the docs locally

Dependencies for building the documentation can be found in `docs/environment.yml`. This is the Conda environment used by ReadTheDocs to build the docs. To build the docs locally, first create the environment, then invoke Sphinx via the makefile:

```shell
# Create the environment
mamba env create --file devtools/conda-envs/docs_env.yaml
# Prepare the current shell session
mamba activate interchange-docs
cd docs
# Build the docs
make html
```

## Pixi

This project uses [Pixi](https://pixi.prefix.dev/latest/) for development and automated testing.

Install Pixi [according to their documentation](https://pixi.prefix.dev/latest/installation/).

Pixi usage boils down to running _tasks_ in _environments_.

### Environments

Several (Pixi) environments are defined. These cover different combinations of Python versions and optional dependencies. List available environments with `pixi workspace environment list`:

```console
$ pixi workspace environment list
Environments:
- default:
    features: default
- py311amber:
    features: py311, test, typing, ambertools, mosdef, default
- py311openeye:
    features: py311, openeye, test, typing, regression_tests, engines, default
- py311examples:
    features: py311, openeye, test, engines, examples, default
- py312amber:
    features: py312, test, typing, ambertools, mosdef, default
- py312openeye:
    features: py312, openeye, test, typing, regression_tests, engines, default
- py312examples:
    features: py312, openeye, test, engines, examples, default
- py313none:
    features: py313, test, typing, mosdef, default
- py313openeye:
    features: py313, openeye, test, typing, regression_tests, engines, default
- py313examples:
    features: py313, openeye, test, engines, examples, default
- dev:
    features: py312, openeye, test, typing, dev, engines, default
- betas:
    features: py311, test, typing, engines, betas, default
```

Each environment can be thought of as a combination of features. Each feature can be thought of as a collection of dependencies neccessary for a particular use case. For example, the feature named `typing` defines `mypy` and `pandas-stubs` as dependencies. Any environment that lists `typing` as a feature will include these packages. List all included features with `pixi workspace feature list`:

```console
$ pixi workspace feature list
Features:
- default:
    dependencies: pip, numpy, setuptools_scm, pydantic, openff-toolkit-base, openff-interchange, rdkit, openmm, python
    tasks: postinstall, run_tests, run_mypy, run_regression_tests, run_examples
- ambertools:
    dependencies: ambertools, numpy
- openeye:
    dependencies: openeye-toolkits
- test:
    dependencies: pytest, pytest-cov, pytest-xdist, pytest-randomly, intermol, nglview
- examples:
    dependencies: mdtraj, nbval, nglview, jax
- typing:
    dependencies: mypy, pandas-stubs
- dev:
    dependencies: pre-commit
- regression_tests:
    dependencies: deepdiff, rich, click
- engines:
    dependencies: gromacs, lammps, numpy
- mosdef:
    dependencies: mbuild-base, foyer, numpy
- betas:
    dependencies: openeye-toolkits, openmm
- py311:
    dependencies: python
- py312:
    dependencies: python
- py313:
    dependencies: python
```

(Note that this view only displays dependencies and not pinned versions. Look at the configuration file for all details.)

### Tasks

List available tasks with `pixi task list`:

```console
$ pixi task list
Tasks that can run on this machine:
-----------------------------------
postinstall, run_examples, run_mypy, run_regression_tests, run_tests
Task  Description
```

Each of these (Pixi) tasks defines some other action, ideally with a descriptive name.

### Running tasks

Run tasks with `pixi run -e <ENVIRONMENT_NAME> <TASK_NAME>`. Note that not all environment-task combinations are valid. See the CI script for combinations that are currently used. Currently it is set up to run `run_examples` and `run_regression_tests` in "examples" environments and `run_mypy` and `run_tests` in other environments.

For example, run examples with Python 3.12 as

```console
$ pixi run -e py312examples run_examples
...
```

or run unit tests with Python 3.11 as

```console
$ pixi run -e py311amber run_tests
...
```

or run `pre-commit` in a development environment:

```console
$ pixi run -e dev pre-commit run -a
...
```
