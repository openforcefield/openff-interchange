# Sharp edges

## Quirks of `from_openmm`

### Modified masses are ignored

The OpenFF toolkit does not support isotopes or modifiying masses from the values defined in the periodic table. In the `Topology` and `Molecule` classes, particles masses are defined only by their atomic number. When topologies are read from OpenMM, the particle mass is ignored and the atomic number of the element is read.

As a consequence, systems with hydrogen mass repartitioning (HMR) applied have HMR un-done (masses are shifted from hydrogens back to their heavy atoms). To re-apply HMR (shift masses back to hydrogens), use the API at export time, likely with a `hydrogen_mass` argument in an export method.

For updates, [search "HMR" in the issue tracker](https://github.com/search?q=repo%3Aopenforcefield%2Fopenff-interchange+hmr&type=issues&s=updated&o=desc) or raise a [new issue](https://github.com/openforcefield/openff-interchange/issues/new/choose).

Keywords: HMR, hydrogen mass repartioning

## Quirks with GROMACS

### Residue indices must be begin at 1

Whether by informal convention or official standard, residue numbers in GROMACS files being at 1. Other tools may start the residue numbers at 0. If a topology contains residue numbers below 1, exporting to GROMACS will trigger an error (though not necessarily while exporting to other formats). A workaround for 0-indexed residue numbers is to simply increment all residue numbers by 1.

For more, see [issue #1007](https://github.com/openforcefield/openff-interchange/issues/1007)

Keywords: GROMACS, residue number, resnum, residue index
