# Sharp edges

## Quirks of charge assignment

### Charge assignment hierarchy

Interchange, following the [SMIRNOFF specification](https://openforcefield.github.io/standards/standards/smirnoff/#partial-charge-and-electrostatics-models), assigns charges to (heavy) atoms with the following priority:

1. **Preset charges**: Look for molecule matches in the `charge_from_molecules` argument
2. **Library charges**: Look for chemical environment matches in the `<LibraryCharges>` section of the force field
3. **Charge increment models**: Look for chemical environment matches in the `<ChargeIncrementModel>` section of the force field
4. **AM1-BCC**: Try to run some variant of AM1-BCC as described by the `<ToolkitAM1BCC>` section of the force field

If charges are successfully assigned using a method, no lower-priority methods in this list are attempted. For example:

* A force field with library charge parameters for peptides (i.e. a biopolymer force field, covering all appropriate residues) will NOT try to call AM1-BCC on a biopolymers.
* If a ligand is successfully assigned preset charges, chemical environment matching of library charges and charge increments will be skipped, as will AM1-BCC.
* If a variant of AM1-BCC (i.e. using something other than AM1 and/or using custom BCCs) is encoded in a `<ChargeIncrementModel>` section, other AM1-BCC implementations will not be called.
* If preset charges are not provided, a force field like OpenFF's Parsley or Sage lines without (many) library charges or charge increments will attempt AM1-BCC on all molecules (except water and monoatomic ions).

After all of these steps are complete and all heavy atoms given partial charges, virtual sites are assigned charges using the values of `charge_increment`s in the virtual site parameters. Because virtual site charge are only described by the force field, using preset charges with virtual sites is discouraged.

### Preset charges

The charges specified by the force field can be overridden by providing molecules with partial charges to the `charge_from_molecules` argument. This may be used to make use of alternate implementations of the appropriate charge generation method, or to provide different charges to the force field. Charges provided via `charge_from_molecules` are called "preset charges" because they are pre-set by the user, rather than computed by the force field. The following restrictions are in place when using preset charges:

* All molecules in the the `charge_from_molecules` list must be non-isomorphic with each other.
* All molecules in the the `charge_from_molecules` list must have partial charges.
* All copies of a molecule in the topology will be parametrized with the charges from an isomorphic molecule from the `charge_from_molecules` list.

Using preset charges with virtual sites is discouraged as it can provide surprising results.

## Quirks of core OpenFF objects

Future refactors may remove the side effects of these quirks, but currently there are some
surprising inconsistencies in some API points between different OpenFF tools.

### Contents of `Interchange.topology` and `Interchange` may not always be in sync

Currently, the `Interchange.topology` attribute is defined by the OpenFF Toolkit's `Topology` object, which is feature-rich in cheminformatics functionality but not designed for MM interoperability. Most importantly, that representation does not include virtual sites (because molecules do not have virtual sites/dummy atoms). As a result, functionality involving virtual sites must go through `Interchange` API points instead of `Interchange.topology`.

For example, `Interchange.topology.get_positions()` will never include positions of virtual sites. To get the positions of a system with virtual sites included, use `Interchange.get_positions()`. The default behavior of `Interchange.positions` is also to return positions without virtual sites, but this may change in the future.

### Existing charges are ignored by default

`Molecule` and `Topology` objects can store partial charges, but these are ignored by default in methods like `Interchange.from_smirnoff`. This is because partial charges in SMIRNOFF force fields are defined by sections in the force field. To override this behavior, use the `charge_from_molecules` argument. Be aware that charges, and as a result most physics, will differ from what's perscribed by the contents of the force field.

## Quirks of `Interchange.combine`

### Electrostatics 1-4 scaling factors may be slightly modified

Amber family force fields historically use a 1-4 scaling factor of 1 / 1.2 (or 5/6). Some older OpenFF force field releases round this to 6 digits (0.833333) but more recent releases round this to 10 digits (0.8333333333). These are not strictly equal and arguably should not be combined, but (in this case only) for easier compatibility this difference is ignored and combination proceeds as if they were both 0.833333333 from the start. If value differ significantly, i.e. 0.5 vs 0.833333, an error is raised as this difference is non-trivial.

### Charges of isomorphic molecules may be overwritten

When isomorphic molecules are found on the `Interchange` objects used in `.combine`, the charges of molecules in the argument object are used. For example, if objects `interchange1` and `interchange2` each contain toluene molecules with different partial charges, the charges from `interchange2` will be used on the object returned by `interchange1.combine(interchange2)`. For more, see [Issue #1075](https://github.com/openforcefield/openff-interchange/issues/1075).

## Quirks of `from_openmm`

### Modified masses are ignored

The OpenFF Toolkit does not support isotopes or atomic masses other than the values defined in the periodic table. In the `Topology` and `Molecule` classes, particles masses are defined only by their atomic number. When topologies are read from OpenMM, the particle mass is ignored and the atomic number of the element is read and used to define the atomic properties.

As a consequence, any hydrogen mass repartitioning (HMR) applied to a system is "un-done" upon import --- mass is shifted back from hydrogens to their heavy atoms. To re-apply HMR (shift masses back to hydrogens), use the appropriate API at export time, typically with an export method's `hydrogen_mass` argument.

For updates, [search "HMR" in the issue tracker](https://github.com/search?q=repo%3Aopenforcefield%2Fopenff-interchange+hmr&type=issues&s=updated&o=desc) or raise a [new issue](https://github.com/openforcefield/openff-interchange/issues/new/choose).

Keywords: OpenMM, HMR, hydrogen mass repartioning

### Force constants of constrained bonds may be lost in conversions

Commonly, OpenMM systems prepared by other tools constrain bonds (i.e. bonds between hydrogen and heavy atoms) and/or use rigid water models. These topological bonds lack force constants, which are required by some other engines even though they do not affect the behavior of the simulation.

For example, consider an OpenMM system prepared, from the OpenMM API, with `ForceField.createSystem(..., constraints=HBonds, rigidWaters=True)` and then imported with `Interchange.from_openmm`. The constrained bonds, including all bonds in waters, don't contain physics parameters (particularly the force constant `k`). These parameters are typically dropped from the corresponding `HarmonicBondForce` when running OpenMM simulations with these sorts of constraints. When exporting this system to GROMACS or another engine that needs these parameters, the export will either crash due to missing parameters or silently write a file with some missing or blank parameters.

For more, see [issue #1005](https://github.com/openforcefield/openff-interchange/issues/1005#issue-2405679510).

Keywords: OpenMM, GROMACS, constraints, bond constraints, rigid water

### Only `ThreeParticleAverageSite` virtual sites are supported

Other virtual site types will raise an error.

### Virtual site exclusions re-created with "parents" virtual site exclusion policy

Non-bonded exclusions involving virtual sites (between virtual sites and heavy atoms or between
virtual sites and virtual sites) are not processed. Instead, they are later re-generated assuming the "parents" exclusion policy as defined in the [SMIRNOFF specification](https://openforcefield.github.io/standards/standards/smirnoff/#virtualsites-virtual-sites-for-off-atom-charges). This should re-create typical exclusions in 4- and 5-site water models but may not be appropriate with highly custom virtual site interactions in larger molecules.

### Virtual sites from multiple sources cannot be mixed

Combining interchanges with virtual sites from multiple sources is not fully-featured. For example, this refers to importing a box of TIP4P-containing solvent from OpenMM with a ligand prepared with SMIRNOFF virtual sites parameters.

### Virtual sites must be listed after heavy atoms each molecule

It's assumed that, in each molecule in an OpenMM topology, all heavy atoms are listed before any virtual sites. This includes the case of all virtual sites being listed after all heavy atoms, i.e. not collated into molecules/residues. There are no community standards around particle ordering, but virtual sites are typically listed after heavy atoms in each molecule or residue.

### Center-of-mass motion remover ignored

If present, an `openmm.CMMotionRemover` "force" is ignored when loading a system. An `Interchange` object does not store this information. Note that exported systems from `Interchange.to_openmm_system` include this force by default.

## Quirks with GROMACS

### Residue indices must begin at 1

Whether by informal convention or official standard, residue numbers in GROMACS files begin at 1. Other tools may start the residue numbers at 0. If a topology contains residue numbers below 1, exporting to GROMACS will trigger an error (though not necessarily while exporting to other formats). A workaround for 0-indexed residue numbers is to simply increment all residue numbers by 1.

For more, see [issue #1007](https://github.com/openforcefield/openff-interchange/issues/1007)

Keywords: GROMACS, residue number, resnum, residue index
