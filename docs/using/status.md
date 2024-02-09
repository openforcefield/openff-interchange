# Capabilities

| Engine | Known bugs | HMR | 4-site water models | 5-site water models | Virtual sites on ligands | Proteins | Nucleic acids | Lipids | Carbohydrates |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| All | | | | | | <td colspan=3>No supported SMIRNOFF force fields |
| OpenMM | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Aopenmm+label%3Abug) | Mostly<sup>2</sup> supported | Supported | Supported | Supported | Supported |
| GROMACS | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Agromacs+label%3Abug) | Mostly<sup>2</sup> supported | Partially supported | Supported | Partially supported | Supported |
| Amber | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Aamber+label%3Abug) | Unsupported | Unsupported<sup>1</sup> | Unsupported<sup>1</sup> | Unsupported<sup>1</sup> | Supported |
| LAMMPS | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Alammps+label%3Abug) | Unsupported | Not supported | Not supported | Not supported | Not tested |

1. Unable to find upstream documentation
2. See caveats in `Interchange.to_openmm` and `Interchange.to_gromacs` docstrings.
