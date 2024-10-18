# Capabilities

| Engine | Known bugs | HMR | Bond constraints | 4-site water models | 5-site water models | Virtual sites on ligands | Proteins | Nucleic acids | Lipids | Carbohydrates |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| OpenMM | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Aopenmm+label%3Abug) | Supported | Supported | Supported | Supported | Supported | Supported |
| GROMACS | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Agromacs+label%3Abug) | Mostly<sup>2</sup> supported | Partially supported | Partially supported | Supported | Partially supported | Supported |
| Amber | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Aamber+label%3Abug) | Unsupported | Partially supported | Unsupported<sup>1</sup> | Unsupported<sup>1</sup> | Unsupported<sup>1</sup> | Supported |
| LAMMPS | [Link](https://github.com/openforcefield/openff-interchange/issues?q=is%3Aissue+is%3Aopen+label%3Alammps+label%3Abug) | Unsupported | Partially supported | Not supported | Not supported | Not supported | Not tested |
| All | | | | | | | <td colspan=3>No available SMIRNOFF force fields |

1. Unable to find upstream documentation
2. See caveats in `Interchange.to_gromacs` docstring.
