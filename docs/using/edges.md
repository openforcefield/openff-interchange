# Sharp edges

## Quirks of core OpenFF objects

Future refactors may remove the side effects of these quirks, but currently there are some
surprising inconsistencies in some API points between different OpenFF tools.

### Contents of `Interchange.topology` and `Interchange` may not always be in sync

Currently, the `Interchange.topology` attribute is defined by the OpenFF Toolkit's `Topology` object, which is feature-rich in cheminformatics functionality but not designed for MM interoperability. Most importantly, that representation does not include virtual sites (because molecules do not have virtual sites/dummy atoms). As a result, functionality involving virtual sites must go through `Interchange` API points instead of `Interchange.topology`.

For example, `Interchange.topology.get_positions()` will never include positions of virtual sites. To get the positions of a system with virtual sites included, use `Interchange.get_positions()`. The default behavior of `Interchange.positions` is also to return positions without virtual sites, but this may change in the future.

### Existing charges are ignored by default

`Molecule` and `Topology` objects can store partial charges, but these are ignored by default in methods like `Interchange.from_smirnoff`. This is because partial charges in SMIRNOFF force fields are defined by sections in the force field. To override this behavior, use the `charge_from_molecules` argument. Be aware that charges, and as a result most physics, will differ from what's perscribed by the contents of the force field.

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

### Virtual site exclusions re-created with "parents" virtual site exclusion policy

Non-bonded exclusions involving virtual sites (between virtual sites and heavy atoms or between
virtual sites and virtual sites) are not processed. Instead, they are later re-generated assuming the "parents" exclusion policy as defined in the [SMIRNOFF specification](https://openforcefield.github.io/standards/standards/smirnoff/#virtualsites-virtual-sites-for-off-atom-charges). This should re-create typical exclusions in 4- and 5-site water models but may not be appropriate with highly custom virtual site interactions in larger molecules.

## Quirks with GROMACS

### Residue indices must begin at 1

Whether by informal convention or official standard, residue numbers in GROMACS files begin at 1. Other tools may start the residue numbers at 0. If a topology contains residue numbers below 1, exporting to GROMACS will trigger an error (though not necessarily while exporting to other formats). A workaround for 0-indexed residue numbers is to simply increment all residue numbers by 1.

For more, see [issue #1007](https://github.com/openforcefield/openff-interchange/issues/1007)

Keywords: GROMACS, residue number, resnum, residue index
