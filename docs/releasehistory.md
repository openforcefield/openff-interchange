# Release History

Releases follow versioning as described in
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

Dates are given in YYYY-MM-DD format.

Please note that all releases prior to a version 1.0.0 are considered pre-releases and many API changes will come before a stable release.

## Current development

* #1332 Introduces the following behavior changes to the private Packmol wrappers:
  * Packmol version 20.15.0 or newer is recommended.
  * Periodic boundary conditions are accounted for when placing molecules if the box is orthorhombic and Packmol version 20.15.0 or newer is installed; this is the minimum version supporting this feature.
    * This is functionally the same as the previous behavior, so no workaround is needed to recover it.
  * The `tolerance` parameter is subtracted from computed box lengths when placing molecules if a Packmol version older than 20.15.0 is installed; this is the previous behavior.
  * The `target_density` is used in box size calculations without modification; previously, box volumes were scaled up by a factor of 1.1.
    * The previous behavior can be restored by passing scaling the `target_density` argument down by a factor of 1.1.

* #1376 Makes `Interchange.topology` not store positions. Use `Interchange.positions` instead.
* #1387 Migrates version handling to `setuptools-scm`.
* #1093 Migrates development and automation to use Pixi.

### Bug fixes

* #1396 Fixes charge ordering in Amber files

### Miscellaneous improvements

* #1407 `get_charge_array` is now defined on `ElectrostaticsCollection` and therefore accessible to subclasses `FoyerElectrostaticsHandler` and `BasicElectrostaticsCollection`. Previously, it was only defined on `SMIRNOFFElectrostaticsCollection`.

## 0.4.9 - 2025-11-06

### Behavior changes

* #1352 `Collection` and its subclasses (likely anything with "Collection" in its name) now define equality by identity.

### Performance improvements

* #1352 Improves speed of unit conversions in many submodules and OpenMM export in particular.

### Miscellaneous improvements

* #1353 and #1355 Add some export performance benchmarks.
* #1368 Fixes small molecule regression tests.

### Documentation improvements

* #1362 Document how to opt in to pseudo-vacuum approximation.
* #1364 Fix some internal cross-references in documentation.

## 0.4.8 - 2025-10-09

### Bug fixes

* #1340 Fixes JSON deserialization issues when charges come from `NAGLCharges` or `ChargeIncrementModel`.
* #1325 Fixes some parameter bookkeepping in `Interchange.combine`, particularly when combining similar objects previously created too many `_DUPLICATE` tags.
* #1334 Improves error reporting when Amber calculates non-numeric energies.
* #1337 Fixes a bug in which errors were sometimes raised when valence parameters were missing but not needed.

### Miscellaneous improvements

* #1327 and #1328 Make small improvements to internal logic in `Interchange.combine`.

## 0.4.7 - 2025-09-11

### Bug fixes

* #1321 OpenFF NAGL is treated as an optional dependency.

## 0.4.6 - 2025-09-02

### Bug fixes

* #1312 Fix handling of `DoubleExponentialVirtualSites`
* #1315 Fixes a bug in which `Interchange.combine` did not properly process constraint distances.

### Maintenance

* #1297 Do not duplicate type annotations in docstrings
* #1304 Automatically format `pyproject.toml`
* #1306 Declare minimum Python version in project metadata

## 0.4.5 - 2025-08-20

### New features

* #1206 Support `<NAGLCharges>` tags in SMIRNOFF force fields.

### Bug fixes

* #1275 Fix JSON deserialization with preset charges

### Documentation improvements

* #1294 Fix `box_shape` docstring.

## 0.4.4 - 2025-07-29

### Behavior changes

* #1273 In `solvate_topology`, ions counts are determined using the SLTCAP method.

### Bug fixes

* #1273 `solvate_topology` now returns a charge-neutral topology.
* #1272 In `pack_box` and other Packmol-wrapping functions, molecules with zero count are skipped.

### Miscellaneous improvements

* #1263 Removes `importlib_metadata` backport when loading plugins.

### Documentation improvements

* #1259 Remove obsolete warnings of experimental/unstable status

## 0.4.3 - 2025-07-11

### New features

* #1174 Support Python 3.13.
* #1216 Type labels can (optionally) be included in LAMMPS files.
* #1219 Adds `SMIRNOFFElectrostaticsCollection.get_charge_array`, which currently only works with systems lacking virtual sites.
* #1220 Adds `Interchange.set_positions_from_gro`.
* #1250 Allow `Interchange.combine` to proceed when cutoffs differ by up to 1e-6 nanometers.

### Behavior changes

* #1194 Drop support for Python 3.10.
* #1137 Internally use `openmm.LangevinMiddleIntegrator`, including in the OpenMM driver.
* #1186 `GROMACSSystem.molecules` is changed from a dictionary to a list of tuples.
* #1192 `Interchange.to_pdb` now uses `ensure_unique_atom_names="residues"` when writing PDB files, matching longstanding behavior of the toolkit. For more flexibility, use the `Interchange.topology` and options provided by `openff.toolkit.Topology.to_file`.
* #1205 Hybrid pair styles, which were never necessary, are no longer used in LAMMPS files.
* #1215 The `topology` argument to `Interchange.from_openmm` is now strictly required. Previously, it could be `None` but would error.
* #1225 Treat asterisks as comments in GROMACS files.

### Bug fixes

* #1186 Fixes an issue in which GROMACS topology files were mangled when molecules were not grouped within a topology.
* #1200 Atoms in improper torsions in LAMMPS files are now ordered correctly (with the central atom first).
* #1200 `ImproperTorsionKey.get_central_atom_index` now returns fot correct atom index (this class stores the central atom first).

### Miscellaneous improvements

* #1243 Use `pyedr` instead of `panedr` for quicker processing of GROMACS energy files.
* #1229 Improve detailed energy reporting.
* #1175 Make the system in the protein-ligand example charge-neutral.
* #1183 Remove warning about use of `Interchange.combine`.
* #1241 Document how to control GROMACS molecule names.
* #1224 Improve errors when parsing GROMACS files.

## 0.4.2 - 2025-02-26

### Behavior changes

* #1153 Changes the electrostatics 1-4 scaling factor of 0.833333 to 0.8333333333 when `Interchange.combine` would otherwise fail.

### New features

* #1120 Read GROMACS topology files split into `.itp` files.

### Performance improvements

* #1126 Fixes some charge caching issues that slowed GROMACS exports.

### Documentation improvements

* #1163 Fixes some Packmol wrapper docstrings.
* #1159 Documents why `tolerance` is removed from box edges in the Packmol wrappers.
* #1155 Documents an edge case with `Interchange.combine` when isomorphic molecules are present with different partial charges.

### Miscellaneous improvements

* #1140 Bumps internal tests to use OpenMM 8.2.
* #1150 Updates which examples are tagged as "experimental."

## 0.4.1 - 2025-01-22

### Behavior changes

* #1133 Methods that return an `openmm.System` object now include a `openmm.CMMotionRemover` "force" within it.

### New features

* #1081 `Interchange.from_openmm` now processes virtual sites, but only `openmm.ThreeParticleAverageSite`s.
* #1053 Logs, at the level of `logging.INFO`, how charges are assigned by SMIRNOFF force fields to each atom and virtual site.
* #1080 Adds support for HMR in OpenMM when virtual sites are present.
* #1119 Adds support for writing GROMACS `.itp` files.

### Bug fixes

* #1070 The `charge_from_molecules` argument must include only molecules that contain partial charges and are non-isomorphic with each other.
* #1070 The `charge_from_molecules` argument as used by the OpenFF Toolkit is handled internally as `molecules_with_preset_charges`.
* #1134 Fixes a bug in which importing OpenMM systems containing only rigid water would crash.
* #1111 Conformers except the first conformer are discarded in packing methods.

### Performance improvements

* #1097 Migrates version handling to `versioningit`, which should result in shorter import times.
* #1122 Improves performance of Interchange.from_smirnoff on polymers.
* #1115 Improves processing some non-bonded parameters via caching common operations.
* #1085 Defers some third-party imports for faster import times.

### Documentation improvements

* #1070 Documents charge assignment hierarchy in the user guide.
* #1105 Documents new Pydantic-compatible type annotations in the user guide.

### Miscellaneous improvements

* #1104 Consolidates configuration files to `pyproject.toml` where possible.
* #1089 Adds regression test for JSON serialization after `Interchange.from_openmm`.

## 0.4.0 - 2024-11-04

### Breaking changes and behavior changes

* Pydantic v2 is now used at runtime:
  * V2 must be installed, but other packages may use the `pydantic.v1` backdoor to access the v1 API.
  * Models containing `Interchange`s cannot also use models from the v1 API.
* `Interchange.to_gromacs` and similar methods now raise an error if no box is defined.
  * Previously, this was a warning.
  * GROMACS dropped support for (proper) vacuum simulations in version 2020 and there are no immediate plans to re-introduce it.
  * Users freqently approximate vacuum simulation with periodic boundary conditions by applying a large box. This has some performance issues and some non-bonded terms likely differ numerically compared to similar implementations in other engines.
* The `topology` attribute of the `Interchange` class is now required.
* The wrapped use of external readers and writers is removed, and with it arguments like `writer` in many `Interchange.to_x` calls.
* Several classes and methods which were deprecated in the 0.3 line of releases are now removed.
* Previously-deprecated examples are removed.
* `ProperTorsionKey` no longer accepts an empty tuple as atom indices.
* Packing functions have been overhauled:
  * Argument `mass_density` to some packing functions has been renamed to `target_density` for consistency and to better reflect its usage.
  * Default densities for Packmol wrapper functions using water as the primary solvent have been lowered to 0.9 g/cc.
  * `target_density` is now a required argument to `solvate_topology_nonwater`.
  * Topologies returned by packing functions have boxes scaled up by 10% in linear dimensions compared to the size implied by the target density.
  * PACKMOL failures are now better reported to the user.
* An error is now raised when HMR would result in an OpenMM particle (aside from virtual sites) having negative (or zero) mass.
* `to_openmm_topology` now adds virtual sites to the end of the topology by default, matching the ordering of particles in `to_openmm_system`. Previous atom ordering can be forced with `collate=True`.

### Documentation improvements

* Documented that existing charges on input topologies are ignored.
* Adds an example of basic Amber usage.

### New features

* `hydrogen_mass` and `ewald_tolerance` are now arguments to `to_openmm_simulation`, passed on to `to_openmm_system`.
* Adds `Interchange.get_positions`, which includes positions of virtual sites by default.
* Adds `Interchange.to_amber`, which writes the parameter/topology, coordinate, and (`sander`) run/input files.
* Improves handling of residue metadata to OpenMM with virtual sites.
* Collections can now be indexed using tuples of atom indices.
* Adds high-level run/input file getters (`Interchange.to_{mdp|lammps_datafile|sander_input}`).

### Bug fixes

* Removes an internal use of the `@experimental` decorator which prevented `Interchange.from_openmm` from being fully removed from an experimental state.
* Fixes a regression in which some `ElectrostaticsCollection.charges` properties did not return cached values.
* Better process atom names in `Interchange.from_openmm`
* Fixes regression tests.

## 0.3.30 - 2024-08

* #1039 Updates support of "cutoff" electrostatics in `.to_openmm` to better reflect what OpenMM supports. Set `"reaction-field"` to force the use of `CutoffPeriodic`, provided the vdW and electrostatic cutoff distances match. The potential/method `"cutoff"` is no longer supported but may be re-added in the future.

## 0.3.29 - 2024-08-01

* #1023 Fixes a bug in which non-bonded parameter lookup sometimes crashed when virtual sites were present.
* #1018 #1019 Improve performance of `Interchange.to_top`, particularly after importing from OpenMM objects.

## 0.3.28 - 2024-07-17

* #991 Fixes the virtual site example for upstream changes in virtual site parameter lookup.
* #1000 Fixes a bug in which molecule IDs were mangled in the LAMMPS export.

## 0.3.27 - 2024-05-29

* #980 Fixes #978 in which some dihedral parameters were lost in `Interchange.from_gromacs`.
* #982 Improves handling of plugins with custom vdW interactions on virtual sites.
* #972 Fixes a bug in which PME electrostatics "cutoffs" were not parsed in `from_openmm`.
* #975 Fixes a bug in which some molecule names were not unique in GROMACS export.
* #987 Supports systems with >99,999 atoms in the PACKMOL wrapper.

## 0.3.26 - 2024-04-16

* #952 Drops support for Python 3.9.
* #853 Better support LJPME in some GROMACS tests.
* #954 Fixes a broken link in plugin documentation.
* #963 Removes pre-set partial charges from host-guest example.
* #956 Adds another helper function for solvation in non-aqueous solvent.
* #946 Adds support for GROMACS's `3fad` virtual sites.
* #918 Improves storage of cosmetic attributes.
* #880 Improves virtual site example notebook.

## 0.3.25 - 2024-03-29

* #947 Fixes a bug in which virtual site parameters with identical SMIRKS would clash.
* #948 Updates the GAFF example.
* #950 Fixes a bug in which atom ordering was often incorrect in GROMACS `.gro` files when virtual sites were present.
* #942 Fixes an errant internally-thrown `DeprecationWarning`.
* #942 Recommends `jupyter-lab` over `jupyer-notebook`.

## 0.3.24 - 2024-03-19

* #925 Adds documentation of upcoming v0.4 changes.
* #933 Fixes #934 in which atom order was sometimes mangled in `Interchange.from_openmm`.
* #932 Fixes #935 in which `KeyError` was sometimes raised after `Interchange.combine`.
* #929 A warning is raised when positions are not passed to `Interchange.from_openmm`.
* #930 Adds `additional_forces` argument to `create_openmm_simulation`.
* #938 An error is raised when non-bonded settings do not match when using `Interchange.combine`.

## 0.3.23 - 2024-03-07

* #923 An error is raised in `Interchange.from_openmm` when the topology and system are incompatible.
* #912 A warning is raised when writing an input/run file (not data file) to an engine that does not implement a switching function described by SMIRNOFF.
* #916 Some internal code paths are re-organized, including removing the `openff.interchange.interop.internal` submodule.
* #916 Improves speed of `Interchange.to_lammps`, particularly for larger systems.
* #920 Fixes a bug in which virtual site exclusions were incorrect when using split non-bonded forces.
* #915 Deprecates `Interchange.__add__` in favor of `Interchange.combine`.
* #897 Improves energy evaluation with LAMMPS when some bonds are constrained.

## 0.3.22 - 2024-02-27

* #912 Fixes a bug in which rigid water geometries were incorrectly written to GROMACS files.
* #909 Fixes a bug in which numerical values such as `scale_14` were lost when parsing JSON dumps.

## 0.3.21 - 2024-02-20

* #906 Fixes a bug in which intramolecular interactions between virtual sites were not properly excluded with OpenMM.
* #901 `Interchange.from_openmm` now requires the `system` argument.
* #903 The Python API of LAMMPS is now internally used for LAMMPS energy calculations.

## 0.3.20 - 2024-02-12

* #891 Adds support for hydrogen mass repartitioning (HMR) in GROMACS export. Note that this implementation never modifies masses in waters and requires the system contains no virtual sites.
* #887 Adds support for hydrogen mass repartitioning (HMR) in OpenMM export. Note that this implementation never modifies masses in waters and requires the system contains no virtual sites.

### 0.3.19 - 2024-02-05

* #867 Tags `PotentialKey.virtual_site_type` with the associated type provided by SMIRNOFF parameters.
* #857 Tags `PotentialKey.associated_handler` when importing data from OpenMM.
* #848 Raises a more useful error when `Interchange.minimize` is called while positions are not present.
* #852 Support LJPME in OpenMM export.
* #871 Re-introduces Foyer compatibility with version 0.12.1.
* #883 Improve topology interoperability after importing data from OpenMM, using OpenFF Toolkit 0.15.2.
* #883 Falls back to `Topology.visualize` in most cases.

### Bugfixes

* #848 Fixes a bug in which `Interchange.minimize` erroneously appended virtual site positions to the `positions` attribute.
* #883 Using `openff-models` 0.1.2, fixes parsing box information from OpenMM data.
* #883 Skips writing unnecessary PDB file during visualization.
* #883 Preserves atom metadata when round-tripping topologies with OpenMM.

### Documentation improvements

* #864 Updates installation instructions.

## 0.3.18 - 2023-11-16

### Bugfixes

* #844 Fixes a bug in which charge assignment caching incorrect charges between similar molecules with different atom orderings.

### Behavior changes

* #845 Adds an exception when unimplemented virtual sites are present while writing to Amber files. Previously this was a silent error producing invalid files.

## 0.3.17 - 2023-11-02

### New Features

* #831 Adds `Interchange.minimize` and an underlying implementation with OpenMM.

### Bugfixes

* #830 #834 Updates versioneer configuration for Python 3.12 compatibility.

### Behavior changes

* #835 Most SMIRNOFF virtual site types are once again implemented in OpenMM with ``openmm.LocalCoordinatesSite`` as it is strictly the only proper option.

## 0.3.16 - 2023-10-18

### New Features

* #827 Adds the `ewald_tolerance` option to `to_openmm`, overriding a default value inherited from old versions of the OpenFF Toolkit.
* #827 Adds `to_openmm_system` methods, which alias existing `to_openmm` methods that create `openmm.System`s. Existing methods are not removed or currently deprecated.

### Behavior changes

* #828 Most virtual site types are implemented in OpenMM as types other than ``openmm.LocalCoordinatesSite`` for better human readability.

### Documentation improvements

* #828 Adds a notebook demonstrating, including visualization and running short simulations, several use cases of SMIRNOFF virtual sites.

### Examples added

* #825 Adds a host-guest example derived from the [SAMPL6 challenge](https://github.com/samplchallenges/SAMPL6).

## 0.3.15 - 2023-10-06

### New Features

* #815 Most SMIRNOFF collections are now available via a public interface (`from openff.interchange.smirnoff import ...`).

### Bugfixes

* #814 Tracks `scale_12` in non-bonded handlers.
* #821 Fixes visualization issues described in #819.
* #816 Ensures virtual sites are added at the _end_ of an `openmm.System`, after _all_ atoms (i.e. not interlaced with molecules).

## 0.3.14 - 2023-09-07

### New Features

* #798 `Interchange.to_gromacs` and related methods now support `BondCharge` virtual sites and four-site water models.

### Bugfixes

* #797 Fixes a bug in which virtual site charge increments were not properly applied.

### Breaking changes

* #797 `SMIRNOFFElectrostaticsCollection` now applies virtual site charge increments.
* #797 Removes `ElectrostaticsCollection.charges_without_virtual_sites`, `FoyerElectrostaticsHandler.charges_without_virtual_sites`, and makes `SMIRNOFFElectrostaticsCollection.charges_without_virtual_sites` private.

### Documentation improvements

* #802 Fixes some typos in the `vdWHandler` down-conversion warning.

## 0.3.13 - 2023-08-22

### New Features

* #791 `Interchange.visualize` can now visualize virtual sites via the `include_virtual_sites` argument.
* #791 `get_positions_with_virtual_sites` now supports more virtual site types.

### Behavior changes

* #791 `BondCharge` virtual sites are now implemented in OpenMM with `openmm.TwoParticleAverageSite`.
* #791 Conventional four-site water models using `DivalentLonePair` virtual sites, like TIP4P and its variants, are now implemented in OpenMM with `openmm.ThreeParticleAverageSite`.

## 0.3.12 - 2023-08-14

### New features

* #782 OpenMM is now an optional dependency at runtime or if using the `openff-interchange-base` conda
  package.

## 0.3.11 - 2023-08-09

### Behavior changes

* #789 Internally use vdWHandler 0.4 when storing SMIRNOFF data and creating OpenMM forces.
* #789 Using plugins that create `openmm.NonbondedForce` now results in `openmm.NonbondedForce.NoCutoff` when the topology is non-periodic and `vdWHandler.nonperiodic_method == "no-cutoff"`

## 0.3.10 - 2023-08-02

### New features

* #781 Adds support for version 0.4 of the SMIRNOFF vdW section.
* #780 Adds compatibility with Pydantic v2, using the existing v1 API.

### Behavior changes

* #778 `Interchange.from_foyer` now infers positions from the input topology, matching the behavior of `Interchange.from_smirnoff`.

### Documentation improvements

* #778 Updates the Foyer migration guide.

## 0.3.9 - 2023-07-25

### Bugfixes

* #775 Makes `Interchange.from_openmm` a class method as originally intended.

## 0.3.8 - 2023-07-14

### Behavior changes

* #667 Clarifies lack of support for (hard) cut-off electrostatics in OpenMM.

### New features

* #763 Optionally adds virtual sites to `Interchange.to_pdb` and `Interchange.visualize` and refactors virtual site position fetching into a common module.

### Bugfixes

* #766 Fixes #765 in which the path to the GROMACS energy file argument was hard-coded in `_parse_gmx_energy`.

### Maintenance

* #759 Improves internal handling of non-bonded settings in OpenMM export.

## 0.3.7 - 2023-06-29

### New features

* #579 Adds support for using the geometric mixing rule in OpenMM export.

### Documentation improvements

* #756 Updates example notebooks using a more consistent structure.
* #754 Fixes a call to `pack_box` in the protein-ligand example.

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

* #357 The `_OFFBioTop` constructor now requires an `mdtraj.Topology` passed through the `mdtop` argument.
* #363 This project is no longer tested on Python 3.7

### Bugfixes

* #351 Fix setting byte order while processing bytes to NumPy arrays
* #354 Fix positions processing in `Interchange.__add__`
* `e176033` Fixes non-bonded energies not being parsed while reporting energies from the OpenMM driver.

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

* #269 OpenEye Toolkits are now used in automated testing by default.
* #281 Refactors the test suite into unit tests, interoperability tests, and energy comparison tests.
* #289 Improves the Amber energy driver.
* #292 Improves some ParmEd conversions.
* #232 Fixes `mypy` and updates its configuration.

## 0.1.0 - 2021-06-30

The is an initial pre-release of the OpenFF Interchange.
