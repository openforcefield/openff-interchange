# Release History

Releases follow versioning as described in
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

Please note that all releases prior to a version 1.0.0 are considered pre-releases and many API changes will come before a stable release.

## 0.2.0 - Development

The v0.2.x line of Interchange targets biopolymer support and consistitutes several significant
changes compared to older versions. Because of this, versions v0.2.0 and newer will not generally be compatible with the 0.1.x line or versions of the OpenFF Toolkit older than v0.11.0.

The default branch has been renamed from `master` to `main`.

## 0.1.4 - 2022-01-11

This pre-release of OpenFF Interchange includes interoperability and documentation improvements.

This release supports Python 3.8 and 3.9; it may be compatible with older and newer versions may but is not guaranteed.

### New features
* #355 Add `Interchange.to_pdb`
* #357 Add more type annotations

### Documentation improvements
* #319 Add Foyer showcase (silica nanoparticle solvated in an organic species)
* #358 Fix `output.md`
* #352 Fix some typos in docstrings

### Breaking changes
* #357 The `_OFFBioTop` constructor now requires an `mdtraj.Topology` passed through the `mdtop` argumment.
* #363 This project is no longer tested on Python 3.7

### Bugfixes
* #351 Fix setting byte order while processing bytes to NumPy arrays
* #354 Fix positions processing in `Interchange.__add__`
* `e176033` Fixes nonbonded energies not being parsed while reporting energies from the OpenMM drver.

## 0.1.3 - 2021-11-12

This pre-release of OpenFF Interchange includes documentation improvements and some reliability and testing improvements.

### New features
* #317 Partially avoids parameter clashes in `Interchange` object combination

### Documentation improvements
* #234 Switch documentation them theme to `openff-sphinx-theme`
* #309 Improves the user guide
* #190 Adds parameter splitting example
* #331 Restores `autodoc_pydantic` sphinx extension

### Bugfixes
* #332 Fixes export of multi-molecule systems to Amber files

### Testing and reliability improvements
* #324 Removes `pmdtest` module
* #327 Skips unavailable drivers instead of erroring out
* #246 Improves exports of non-bonded settings to Amber files
* #333 Makes beta/RC tests run automatically


## 0.1.2 - 2021-10-26

This pre-release of the OpenFF Interchange adds preliminary support for exporting to some file formats used by the Amber suite of biomolecular simulation programs and some support for conversions from InterMol objects. Stability, reliability, and feature completeness of these new features is not guaranteed - please report bugs or any surprising behavior.

### Features added
* #310 Adds functions that run all energy drivers at once. See `openff/interchange/drivers/all.py` for details.
* #312 Adds conversion from InterMol `System` objects.
* #316 Adds an experimental GROMACS parser.
* #230 Adds experimental exports to some Amber files (`.inpcrd` and `.prmtop`).

### Bug Fixes
* #308 Fixes a bug involving duck-types NumPy types.
* #322 Fixes a bug in which the Amber driver would not work with some mainline OpenFF force fields.


## 0.1.1 - 2021-09-13

This pre-release of the OpenFF Interchange adds preliminary support for virtual sites and bond order-based parameter interpolation. Stability and reliability with these new features is not guaranteed - please report bugs or any surprising behavior.

**Note**: This release is not compatible with versions of OpenMM older than 7.6.

### Features added
* #252 Improves error handling in cases of unassigned valence terms.
* #228 Adds support for bond-order based interpolation of harmonic bond parameters.
* #263 Adds support for bond-order based interpolation of periodic torsion parameters.
* #244 Adds preliminary support for internally storing virtual sites following the SMIRNOFF specification.
* #253 Adds support for virtual sites modifying partial charges via charge increments.
* #248 Adds  preliminary support exporting systems with virtual sites to GROMACS.
* #268 Adds  preliminary support exporting systems with virtual sites to OpenMM.
* #298 Adds `PotentialHandler.set_force_field_parameters`
* #300 Adds a GROMACS `.gro` file reader.

### Behavior changed
*  #298 Refactors `PotentialHandler.get_mapping` to use `PotentialKey` objects as keys instead of `Potential` objects.

### Documentation improvements
* #267 Adds docstrings for most functions and classes in the source code.
* #285 Adds an example using a SMIRNOFF force field with a liquid-phase mixture of organic compounds.
* #286 Updates the README file.
* #271 Adds automatic API documentation via `autosummary`.

### Testing and reliability improvements
* #269 OpenEye toolkits are now used in automated testing by default.
* #281 Refactors the test suite into unit tests, interoperability tests, and energy comparison tests.
* #289 Improves the Amber energy driver.
* #292 Improves some ParmEd conversions.
* #232 Fixes `mypy` and updates its configuration.


## 0.1.0 - 2021-06-30

The is an initial pre-release of the OpenFF Interchange.
