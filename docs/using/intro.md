# Introduction

OpenFF Interchange is a Python package developed by the Open Force Field
Initiative for storing, manipulating, and converting molecular mechanics data.
The package is oriented around the [`Interchange`] class, which stores
a molecular mechanics system and provides methods to write the system out in
numerous formats.

An `Interchange` contains a fully parameterized molecular system with all the
information needed to start a simulation. This includes the force field, box
vectors, positions, velocities, and a topology containing individual molecules
and their connectivity. For most users, Interchange forms the bridge between
the OpenFF ecosystem and their simulation software of choice; users describe
their system with the [OpenFF Toolkit] and then parameterize it with Interchange.

An `Interchange` stores parameters in handlers that link the topology to the
force field. This allows changes in the force field to be reflected in the
`Interchange` immediately. This is useful for iteratively tweaking parts of a
force field without having to recompute expensive parts like the charges.

Once the Interchange is created and parameterized, it can be exported as
simulation-ready input files to a number of molecular mechanics software packages,
including Amber, OpenMM, GROMACS, and LAMMPS.

:::{mermaid}
---
alt: "Flowchart describing the construction and use of an Interchange (See textual description below)"
align: center
---
flowchart LR
    OFFXML
    SMILES/SDF/PDB
    BoxVecs[Box vectors]
    Positions
    Velocities
    subgraph tk [openff.toolkit]
        Molecule([Molecule])
        ForceField([ForceField])
    end
    subgraph int [openff.interchange]
        Interchange[(Interchange)]
        FromSmirnoff[["from_smirnoff()"]]
    end
    GMX{{GROMACS}}
    CHARMM{{CHARMM}}
    Amber{{Amber}}
    OpenMM{{OpenMM}}
    LAMMPS{{LAMMPS}}

    style tk fill:#2f9ed2,color:#fff,stroke:#555;
    style int fill:#ee4266,color:#fff,stroke:#555;

    classDef default stroke:#555;

    classDef code font-family:cousine,font-size:11pt,font-weight:bold;
    class FromSmirnoff,Molecule,ForceField,Interchange,tk,int code

    OFFXML --> ForceField --> FromSmirnoff
    SMILES/SDF/PDB --> Molecule --> FromSmirnoff
    FromSmirnoff --> Interchange
    BoxVecs -..-> Interchange
    Positions -..-> Interchange
    Velocities -..-> Interchange
    Interchange --> Amber
    Interchange --> OpenMM
    Interchange -.-> GMX
    Interchange -.-> CHARMM
    Interchange -.-> LAMMPS
:::


## Interchange's goals

OpenFF Interchange aims to provide a robust API for producing identical,
simulation-ready systems for all major molecular mechanics codes with the Open
Force Field software stack. Interchange aims to support systems created with
the [OpenFF Toolkit], which can be converted to `Interchange` objects by
applying a SMIRNOFF force field from the Toolkit. The
`Interchange` object can then produce input files for downstream molecular
mechanics software suites. At present, it supports Amber and OpenMM. GROMACS,
and LAMMPS support is in place but experimental, and support for CHARMM is
planned.

By design, Interchange supports extensive chemical information about the target
system. Downstream MM software generally requires only the atoms present in the
system and the parameters for their interactions, but Interchange additionally
supports chemical information like their bond orders and partial charges. These
data are not present in the final output, but allow the abstract chemical
system under study to be decoupled from the implementation of a specific
mathematical model. This allows Interchange to easily switch between different
force fields for the same system, and supports a simple workflow for force
field modification.

Converting in the reverse direction is a long term goal of the project.

(interchange_units)=
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
[ParmEd]: https://parmed.github.io/ParmEd/html/index.html
