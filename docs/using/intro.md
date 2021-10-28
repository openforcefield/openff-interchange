# Introduction

OpenFF Interchange is a Python package developed by the Open Force Field
Initiative for storing, manipulating, and converting molecular mechanics data.
Most importantly, the package provides the [`Interchange`] class, which stores
a molecular mechanics system and provides methods to write the system out in
numerous formats.

## Interchange's goals

OpenFF Interchange aims to provide a robust API for producing identical,
simulation-ready systems for all major molecular mechanics codes with the Open
Force Field software stack. Interchange aims to support systems created with
the [OpenFF Toolkit], which can be converted to `Interchange` objects by
applying a SMIRNOFF force field from the Toolkit or a [Foyer] force field. The
`Interchange` object can then produce input files for downstream molecular
mechanics software suites. At present, it supports GROMACS, LAMMPS, and OpenMM,
and support for Amber and CHARMM is planned.

By design, Interchange supports extensive chemical information about the target
system. Downstream MM software generally requires only the atoms present in the
system and the parameters for their interactions, but Interchange additionally
supports chemical information like their bond orders and partial charges. These
data are not present in the final output, but allow the abstract chemical
system under study to be decoupled from the implementation of a specific
mathematical model. This allows Interchange to easily switch between different
force fields for the same system, and supports a simple workflow for force
field modification.

Converting in the reverse direction is not a goal of the project; Interchange is
not intended to provide general conversions between molecular mechanics codes.
This is because individual MM codes each represent the same chemical system in
different ways, and the information, especially metadata, required to safely
and robustly convert is often not present in simulation input files. If
arbitrary conversion is required, consider [ParmEd].

## Units in Interchange

As a best practice, Interchange always associates explicit units with numerical
values. Units are tagged using the [`openff-units`] package, which provides
numerical types associated with commonly used units and methods for
ergonomically and safely converting between units. However, the Interchange API
accepts values with units defined by the [`openmm.units`] or [`unyt`] packages,
and will automatically convert these values to the appropriate unit to be
stored internally. If raw numerical values without units are provided,
Interchange assumes these values are in the correct unit. Explicitly defining
units helps minimize mistakes and allows the computer to take on the mental
load of ensuring the correct units, so we highly recommend it.

Except where otherwise noted, Interchange uses a nm/ps/K/e/Da unit system
commonly used in molecular mechanics software. This forms a coherent set of
units compatible with SI:

| Quantity        | Unit            | Symbol |
|-----------------|-----------------|--------|
| Length          | nanometre       | nm     |
| Time            | picosecond      | ps     |
| Temperature     | Kelvin          | K      |
| Electric charge | electron charge | e      |
| Mass            | Dalton          | Da     |
| Energy[^drvd]   | kilojoule/mol   | kJ/mol |

[^drvd]: Derived unit

[`Interchange`]: openff.interchange.components.interchange.Interchange
[OpenFF Toolkit]: https://github.com/openforcefield/openff-toolkit
[`openff-units`]: https://github.com/openforcefield/openff-units
[`openmm.units`]: http://docs.openmm.org/latest/api-python/app.html#units
[`unyt`]: https://github.com/yt-project/unyt
[Foyer]: https://github.com/mosdef-hub/foyer
[ParmEd]: https://parmed.github.io/ParmEd/html/index.html
