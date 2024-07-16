# Sharp edges

## Quirks of `from_openmm`

### Modified masses are ignored

The OpenFF Toolkit does not support isotopes or modifiying masses from the values defined in the periodic table. In the `Topology` and `Molecule` classes, particles masses are defined only by their atomic number. When topologies are read from OpenMM, the particle mass is ignored and the atomic number of the element is read and used to define the atomic properties.

As a consequence, systems with hydrogen mass repartitioning (HMR) applied have HMR "un-done" (masses are effectively shifted from hydrogens back to their heavy atoms). To re-apply HMR (shift masses back to hydrogens), use the API at export time, likely with a `hydrogen_mass` argument in an export method.

For updates, [search "HMR" in the issue tracker](https://github.com/search?q=repo%3Aopenforcefield%2Fopenff-interchange+hmr&type=issues&s=updated&o=desc) or raise a [new issue](https://github.com/openforcefield/openff-interchange/issues/new/choose).

Keywords: OpenMM, HMR, hydrogen mass repartioning

### Force constants of constrained bonds may be lost in conversions

Commonly, OpenMM systems prepared by other tools constrain bonds (i.e. bonds between hydrogen and heavy atoms) and/or use rigid waters. These (topological) bonds lack force constants, which may be required by other engines.

For example, consider an OpenMM system prepared, from the OpenMM API, with `ForceField.createSystem(..., constraints=HBonds, rigidWaters=True)` and then imported with `Interchange.from_openmm`. The constrained bonds (including all bonds in waters) don't contain physics parameters (force constant `k` and equilibrium bond length). (These parameters are allowed to exist in a `HarmonicBondForce`, and sometimes they do, but there's also no reason for them to be defined when running OpenMM simulations with these constraints, so they're usually dropped.) When exporting this system to GROMACS or another engine that needs these parameters, the export will either crash due to missing parameters or silently write a file with some missing or blank parameters.

For more, see [issue #1005](https://github.com/openforcefield/openff-interchange/issues/1005#issue-2405679510).

Keywords: OpenMM, GROMACS, constraints, bond constraints, rigid water

## Quirks with GROMACS

### Residue indices must be begin at 1

Whether by informal convention or official standard, residue numbers in GROMACS files begin at 1. Other tools may start the residue numbers at 0. If a topology contains residue numbers below 1, exporting to GROMACS will trigger an error (though not necessarily while exporting to other formats). A workaround for 0-indexed residue numbers is to simply increment all residue numbers by 1.

For more, see [issue #1007](https://github.com/openforcefield/openff-interchange/issues/1007)

Keywords: GROMACS, residue number, resnum, residue index
