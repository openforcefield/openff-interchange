# Development guidelines

This is intended to serve as a living document. If something looks out of date, please submit a PR to update it.

For topics or details not specified, refer to the [development guidelines]( https://open-forcefield-toolkit.readthedocs.io/en/latest/developing.html) of the OpenFF Toolkit.

## Supported Python versions

Generally, follow [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html). This means that currently Python 3.7-3.9 are supported as of the inception of this document (February 2021). No effort needs to be made to support older versions (Python 2 or 3.6 or earlier) or newer versions that are not well-supported by the PyData stack.

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

A sample configuration file (`.pre-commit-config.yaml`) is commited to the repo.

First, install `pre-commit`

```
conda install pre-commit -c conda-forge  # also available via pip
```

Then, install the pre-commit hooks (note that it installs the linters into an isolated virtual environment, not the current conda environment):

```
pre-commit install
```

Optionally update the hooks:

```
pre-commit autoupdate.
```

Hooks will now run automatically before commits. Once installed, it should run in a few seconds.
