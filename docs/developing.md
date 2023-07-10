# Development guidelines

This is intended to serve as a living document. If something looks out of date, please submit a PR to update it.

For topics or details not specified, refer to the [development guidelines]( https://open-forcefield-toolkit.readthedocs.io/en/latest/developing.html) of the OpenFF Toolkit.

## Supported Python versions

Generally, follow [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html). This means that currently Python 3.8-3.9 are supported as of the last update of this document (January 2022). No effort needs to be made to support older versions (Python 2 or 3.7 or earlier) or newer versions that are not well-supported by the [PyData](https://pydata.org) stack.

The last release with support for Python 3.7 wass v0.1.3.

## Style

Style is enforced with automated linters that run in CI. See `.github/workflows/lint.yaml` for the checks that take place, each of which can be run locally. Your favorite IDE probably includes good support for most of these linters.

### Linters

* `black`: Automated, deterministic, and consistent code formatting.
* `isort`: Sorts imports.
* `flake8`: Catches many iffy practices before they can grow into bugs. Check the action to see what plugins are used.
* `pyupgrade`: Automatically update code for features and syntax of newer Python versions.

### Type-checking

Type hints are **optional** but encouraged. Many optional flags are passed to `mypy`; check the action for a recommended invocation. Check `setup.cfg` for some configuration, mostly ignoring libraries that do not have support for type-checking.

### Pre-commit

The [`pre-commit`](https://pre-commit.com/) tool can be used to automate some or all of the style checks.
It automatically runs other programs ("hooks") when you run `git commit`. It errors out and aborts the commit if any hooks fail.

Note that tests (too slow) and type-checking (weird reasons) are not run by `pre-commit`. You should still manually run tests before commiting code.

This project uses `pre-commit ci`, a free service that enforces style on GitHub using the `pre-commit` framework.

The configuration file (`.pre-commit-config.yaml`) is commited to the repo. This file speicies the configuration that `pre-commit.ci` bots will use and also any local installations. Note that because this file is checked out in the repo, all developers therefore use the same pre-commit hooks (as will the `pre-commit.ci` bots).

First, install `pre-commit`

```console
$ conda install pre-commit -c conda-forge
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
$ pre-commit autoupdate.
Updating https://github.com/pre-commit/pre-commit-hooks ... already up to date.
Updating https://github.com/asottile/add-trailing-comma ... already up to date.
Updating https://github.com/psf/black ... already up to date.
Updating https://github.com/PyCQA/isort ... already up to date.
Updating https://github.com/PyCQA/flake8 ... already up to date.
Updating https://github.com/asottile/pyupgrade ... already up to date.
Updating https://github.com/pycqa/pydocstyle ... already up to date.
Updating https://github.com/econchick/interrogate ... already up to date.
Updating https://github.com/asottile/blacken-docs ... already up to date.
Updating https://github.com/jumanjihouse/pre-commit-hooks ... updating 2.1.6 -> 3.0.0.
Updating https://github.com/nbQA-dev/nbQA ... already up to date.
Updating https://github.com/kynan/nbstripout ... already up to date.
```

Hooks will now run automatically before commits. Once installed, it should run in a few seconds.

If hooks are installed locally, all linting checks in CI should pass. If hooks are not installed locally or are significatnly out of date, a `pre-commit.ci` bot may commit directly to a PR to make these fixes.

## Documentation

Interchange is documented with Sphinx and hosted by ReadTheDocs at <https://openff-interchange.readthedocs.io>. The documentation is built and served by ReadTheDocs automatically for every pull request --- please check that your changes are documented appropriately!

Interchange uses Autosummary to automatically document the entire public API from the code's docstrings. Docstrings should be written according to the [NumPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html). By default, all modules and members except those beginning with an underscore are included in the API reference. Additional modules such as tests can be excluded from the reference by listing them in the `autosummary_context["exclude_modules"]` variable in `docs/conf.py`.

This is implemented by including a `:recursive:` Autosummary directive in `docs/index.md` and a customised module template `docs/_templates/autosummary/module.rst`. This template produces neatly segmented, complete documentation with a coherent navigation structure.

The remaining parts of the documentation are written in [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/).

### Building the docs locally

Dependencies for building the documentation can be found in `docs/environment.yml`. This is the Conda environment used by ReadTheDocs to build the docs. To build the docs locally, first create the environment, then invoke Sphinx via the makefile:

```shell
# Create the environment
conda env create --file devtools/conda-envs/docs_env.yaml
# Prepare the current shell session
conda activate interchange-docs
cd docs
# Build the docs
make html
```
