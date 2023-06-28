# Release History

Releases follow versioning as described in
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

Please note that all releases prior to a version 1.0.0 are considered pre-releases and many API changes will come before a stable release.

## Current development

### New features

* #579 Adds support for using the geometric mixing rule in OpenMM export.

## 0.3.6 - 2023-06-20

## Behavior changes

* #748 Resolves #747 in which exceptions inherited from `BaseException` against recommended practice. Exceptions now inherit from `InterchangeException` which itself inherits from `Exception`.
* #707 Overhauls the `openff.interchange.components._packmol` module.

### New features

* #731 Adds support for non-rectangular boxes in GROMACS export.
* #707 Improves error handling when attempting to export non-rectangular boxes to Amber and LAMMPS.

## 0.3.5 - 2023-06-14

### New features

* #725 Adds `Interchange.to_openmm_simulation`.

### Bugfixes

* #724 Fixes #723 in which some parameters in GROMACS files were incorrectly written.
* #728 Fixes #719 in which GROMACS coordinate files were written incorrectly when containing more than 100,000 atoms.
* #741 Improves JSON (de)serialization, particularly while parsing `Collection`s.
* #746 Fixes #745 in which `get_amber_energies` did not properly turn off the switching function.
* #746 Fixes #736 in which `get_openmm_energies` ignored `openmm.RBTorsionForce`.

### Documentation improvements

* #744 Removes binder links.

## 0.3.4 - 2023-05-14

### Bugfixes

* #721 Fixes #720 in which units were not checked when writing `[ settles ]` in GROMACS files.

## 0.3.3 - 2023-05-08

* #703 Clarifies the experimental state of some features, which require opt-in to use.

### Behavior changes

* #706 Updates `pack_box` to return a `Topology`.
* #716 Removes InterMol and ParmEd conversions, which were untested and not part of the public API.

### Bugfixes

* #705 Fixes #693 in which 1-4 vdW interactions were not scaled when using `Interchange.to_openmm(combine_nonbonded_forces=False)`
* #702 Fixes #701 in which `Interchange` was needlessly re-exported in the high-level module.

### Documentation improvements

* #673 Adds a vectorized representation example.
* #703 Adds a section to the user guide on accessing experimental functionality.
* #706 Updates the mixed solvent example.
* #708 Updates the protein-ligand example.
* #708 Updates the ligand-in-water example.

## 0.3.2 - 2023-05-02

### Behavior changes

* #677 Replaces `.constraints` with `.potentials` in constraint collections.
* #677 Unifies parameters in Ryckaert-Bellemans torsions around lowercase values (`c0`, `c1`, etc.).

### New features

* #671 Adds `Interchange.to_gromacs` which writes both GROMACS topology and coordinate files at once.
* #677 Improves support for Ryckaert-Bellemans torsions in parsers, writers, and drivers.
* #681 Ports a PACKMOL wrapper from OpenFF Evaluator.
* #692 Tags some features as experimental, requiring opt-in to access.
* #697 Wires `add_constrained_forces` argument through `Interchange.to_openmm`.
* #697 Wires `combine_nonbonded_forces` argument through `get_summary_data` and `get_all_energies`.

### Bugfixes

* #680 Fixes #678 in which, in some cases, text wrapping in Amber files was mangled.
* #680 Fixes #679 in which atom exclusion lists in Amber files were written incorrectly.
* #685 Fixes #682 in which some 1-4 interactions in Amber files were counted incorrectly.
* #695 Fixes #694 in which systems with no electrostatics did not check for plugins.

## 0.3.1 - 2023-04-19

### Behavior changes

* #639 Drops support for Python 3.8, following [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table).
* #635 Moves and re-organizes the contents of `openff.interchange.interop.internal.gromacs` to a new submodule `openff.interchange.interop.gromacs`.
* #662 Moves tests and un-tested modules from the public API to pseudo-private.

### Bugfixes

* #655 Fixes #652 by avoiding writing a blank `RESIDUE_LABEL` section in Amber files.

### New features

* #635 Adds a dedicated class `GROMACSSystem` to represent GROMACS state.
* #635 Adds parsing and exporting between GROMACS files and `GROMACSSystem` objects.
* #651 Adds support for `SMIRNOFFCollection` plugins that depend on multiple `ParameterHandler`s.
* #654 Adds a module `openff.interchange.common` containing base classes for different types of `Collection`s.
* #659 Improves testing for `from_openmm`.
* #649 Removes the use of `pkg_resources`, which is deprecated.
* #660 Moves the contents of `openff.interchange.components.foyer` to `openff.interchange.foyer` while maintaining existing import paths.
* #663 Improves the performance of `Interchange.to_prmtop`.
* #665 Properly write `[ settles ]` directive in GROMACS files.

## 0.3.0 - 2023-04-10

### Behavior changes

* #566 Refactors `EnergyReport` to more explicitly handle comparisons.
* #583, #588, #603 Change some code paths of internal objects.
  * `PotentialHandler` is deprecated for `Collection`.
  * `Interchange.handlers` is deprecated for `Interchange.collections`.
  * `PotentialHandler.slot_map` is deprecated for `Collection.key_map`.
  * Classes found in `openff.interchange.components.smirnoff` are now in `openff.interchange.smirnoff`.
  * Some arguments with `handler` in their names are replaced with `collection`.
* #583 Refactors the internals of `Interchange.from_smirnoff` away from the chain-of-responsibility pattern.
* #601 Groups GROMACS improper torsion energies in the `"Torsions"` key.

### Bugfixes

* #593 Fixes #592 in which OpenMM exports fail to create multiple non-bonded forces without a vdW handler.
* #601 Fixes #600 in which some parameters were rounded to 6 digits in `Interchange.to_top`.
* #598 Fixes #597 in which residue names were incorrectly written to Amber files for single-residue systems.
* #618 Fixes #616 in which positions of multiple molecules were mangled with `to_openmm_positions`.
* #618 Fixes #617 in which the return type of `to_openmm_positions` was inconsistent.
* #582 Allows OpenMM version to change in regression tests.
* #622 Fixes passing some settings to OpenMM's GBSA forces.

### New features

* #569 Allows for quick importing `from openff.interchange import Interchange`.
* #558 Removes a safeguard against a long-since fixed toolkit bug when looking up `rmin_half`.
* #574 Adds more specific subclasses of `TopologyKey`.
* #589 For convenience, per-parameter variables are now stored on `SMIRNOFFCollection.potential_parameters` and its subclasses.
* #591 Adds custom `SMIRNOFFCollection`s via a plugin interface.
* #614 Adds support for GBSA parameters in SMIRNOFF force fields.
* #586 #591 #605 #609 #613 Support custom SMIRNOFF sections with OpenMM.

### Documentation improvements

* #634 Improves documentation for SMIRNOFF plugins.
* #610 Adds duplicate documentation on how to covert to OpenMM-styled unit/quantity objects.
* #699 Updates the project's README.

### Maintenance

* #400 Adds Python 3.10 to testing.
* #561 #564 #572 #577 #580 #581 #590 #625 Update `pre-commit` configuration.
* #562 Removes the use of `python setup.py`.
* #565 Removes duplicate code in favor of `openff-models`.
* #575 #576 Avoid installing old, incompatible versions of `jax` and/or `jaxlib`.
* #578 Updates some type annotations.
* #619 Removes duplicate OpenMM virtual site construction code.

## 0.2.3 - 2022-11-21

### Behavior changes

* #554 `Interchange.to_openmm` now uses `combine_nonbonded_forces=True` by default.

### New features

* #534 An `openmm.Platform` can be specified as an argument to the OpenMM driver.

### Documentation improvements

* #553 Adds a solvation example.

### Bugfixes

* #545 List the central atom first in CVFF style dihedrals in LAMMPS export
* #551 Use `Interchange.box` to define periodicity when exporting to PDB files.

## 0.2.2 - 2022-10-12

This pre-release of Interchange includes improvements in metadata in the Amber export.

### Behavior changes

* #536 Use atom names provided by the toolkit, or element symbols if not provided, in Amber export

### Bugfixes

* #539 Fix case of single-molecule export to Amber with ambiguous residue information

### Examples added

* #533 Add experimental example using `openmmforcefields` to generate ligand parameters
* #524 Add experimental example using `from_openmm` to import a system prepared with OpenMM tools

## 0.2.1 - 2022-09-02

This pre-release of Interchange includes performance improvements in exporters.

### Performance improvements

* #519 Improve runtime of `Interchange.to_top` by bypassing JAX broadcasting
* #520 Improve runtime of `Interchange.to_top` by using a set to track constrained atom pairs

### Behavior changes

* #519 Exports to array representations no longer use `jax.numpy` by default

## 0.2.0 - 2022-08-29

The 0.2.x line of Interchange targets biopolymer support alongside version 0.11.0 of the OpenFF Toolkit.
Due to the scope of changes, versions 0.2.0 and newer will not generally be compatible with the 0.1.x line or versions of the OpenFF Toolkit less than 0.11.0.
In lieu of a changelog entry for this release, **below is a brief summary of the current capabilities of Interchange.**
Future releases will continue with conventional changelog entries.

Imports from OpenFF Toolkit objects:

* `Interchange.from_smirnoff`, consuming SMIRNOFF force fields and OpenFF `Topology` objects, including
  * WBO-based bond order interpolation of valence parameters
  * Virtual sites
* See the [Molecule Cookbook](https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html) for information on preparing molecule inputs and [`Molecule`](https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule) API docs for more information.

Imports from MoSDeF objects:

* `Interchange.from_foyer`
Exports to OpenMM objects:

* `Interchange.to_openmm` for `openmm.System` objects
* `Interchange.to_openmm_topology` for `openmm.app.Topology` objects

Exports to GROMACS files:

* `Interchange.to_top` for `.top` topology files
* `Interchange.to_gro` for `.gro` coordinate files

Exports to AMBER files (EXPERIMENTAL):

* `Interchange.to_prmtop` for `.prmtop` parameter/topology files
* `Interchange.to_inpcrd` for `.inpcrd` coordinate files

Exports to LAMMPS files (EXPERIMENTAL):

* `Interchange.to_lammps` for `.data` data files

Exports to JAX arrays:

* Call `.get_force_field_parameters` or `.get_system_parameters` on a `PotentialHandler` subclass to get a [JAX array](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html) of force field or "system" parameters associated with that handler.

Known issues and limitations:

* Residue hierarchy information may not be preserved in GROMACS and AMBER exports
* Some operations are slow and not yet optimized for performance, including
  * `Interchange.from_smirnoff`, particularly for large systems with many valence terms
  * `Interchange.to_top`, particularly for systems with polymers, including proteins

During development of this release, the default branch has been renamed from `master` to `main`.

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

* #298 Refactors `PotentialHandler.get_mapping` to use `PotentialKey` objects as keys instead of `Potential` objects.

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
