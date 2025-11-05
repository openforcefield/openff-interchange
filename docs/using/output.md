# Exporting to other software

`Interchange` provides several methods to produce input data for other software. Note that none of these methods write out all the information stored in an `Interchange`; they support a design where the principle source of truth is the rich chemical information in the `Interchange` object, and exported files are tools to perform some operation.

(sec-mdconfig)=
## Run control/config files

SMIRNOFF force fields include several parameters that many MD engines consider to be run configuration options rather than force field parameters. These values are essential for accurately simulating output from Interchange, but they are configured in the same files that are used for general control of simulation runtime behavior. As a result, Interchange cannot simply provide complete versions of these files. Instead, Interchange writes stub versions of MD engine run input files. These files must be modified and completed before they can be used to run a simulation.

## General purpose

An [`Interchange`] can be written out as the common PDB structure format with the [`Interchange.to_pdb()`] method:

```python
interchange.to_pdb("out.pdb")
```

## GROMACS

Once an [`Interchange`] object has been constructed, the `.gro`, `.top`, and `.mdp` files can be written using [`Interchange.to_top()`], [`Interchange.to_gro()`], and [`Interchange.to_mdp()`]:

```python
interchange.to_gro("mysim.gro")
interchange.to_top("mysim.top")
interchange.to_mdp("mysim_pointenergy.mdp")
```

The [`Interchange.to_gromacs()`] convenience method produces all three files in one invocation:

```python
interchange.to_gromacs("mysim")  # Produces the same three files
```

Note that the MDP file generated is configured for a single-point energy calculation and must be modified to run other simulations.

### ITP files

By default, the topology is written to as a monolithic file, which can be large. To split this into separate files with the `#include "molecule.itp"` convention, use `monolithic=False`. This produces a functionally equivalent topology file which splits non-bonded interactions into a file `mysim_nonbonded.itp` and each molecule's parameters in a separate file named according to the `Molecule.name` attribute.

### Molecule names

Molecule names in GROMACS files (both in `[ moleculetype ]` and `[ molecules ]` directives) are determined based on the value of the `Molecule.name` attribute of each molecule in the topology. The toolkit allows setting this attribute to any string.

The default value of `Molecule.name` is `None`, which is not suitable for GROMACS files. In this case, Interchange will attempt to generate unique molecule names on the fly. These may look like `MOL_0`, `MOL_1`, potentially incrementing to larger numbers.

## LAMMPS

An [`Interchange`] object can be written to LAMMPS data and run input files with [`Interchange.to_lammps()`]

```python
interchange.to_lammps("data")  # Produces `data.lmp` and `data_pointenergy.in`
interchange.to_lammps(
    "data", include_type_labels=True
)  # includes LAMMPS type labels in `data.lmp`
```

Note that the generated run input file will run a single-point energy calculation and should be modified for the desired simulation.

LAMMPS does not implement a switching function as [commonly used](https://openforcefield.github.io/standards/standards/smirnoff/#vdw) by SMIRNOFF force fields, so these force fields will produce different results in LAMMPS than in OpenMM or GROMACS.

## OpenMM

An [`Interchange`] object can be converted to an `openmm.System` object with [`Interchange.to_openmm()`].

```python
openmm_sys = interchange.to_openmm()
```

By default, this will separate non-bonded interactions into several different `openmm.Force` objects. To combine everything into a single `openmm.NonbondedForce`, use the `combine_nonbonded_forces=True` argument.

The accompanying OpenMM topology can be constructed with the [`Topology.to_openmm()`] method:

```python
openmm_top = interchange.topology.to_openmm()
```

Recall that all unit-bearing attributes within `Interchange` objects are `openff.units.Quantity` objects, which can be converted out to `openmm.unit.Quantity` objects via their `.to_openmm()` method. For example:

```python
openmm_positions: openmm.unit.Quantity = interchange.positions.to_openmm()
openmm_box: openmm.unit.Quantity = interchange.box.to_openmm()
```

If virtual sites are present in the system, they will all be placed at the end of the system and the topology. Optionally, virtual sites can collated within molecules in the topology, associated with the last residue in each molecule. In this case, all virtual sites are still placed at the end of the system. To do this, use `collate=True` as an argument to `Interchange.to_openmm_topology`. For discussion, see [issue #1049](https://github.com/openforcefield/openff-interchange/issues/1049).

## Amber

An `Interchange` object can be written to Amber parameter/topology, coordinate, and SANDER run input files with [`Interchange.to_prmtop()`], [`Interchange.to_inpcrd()`], and [`Interchange.to_sander_input()`]:

```python
interchange.to_prmtop("mysim.prmtop")
interchange.to_inpcrd("mysim.inpcrd")
interchange.to_sander_input("mysim_pointenergy.in")
```

The [`Interchange.to_amber()`] convenience method produces all three files in one invocation:

```python
interchange.to_amber("mysim")  # Produces the same three files
```

Note that the input file generated is configured for a single-point energy calculation with sander and must be modified to run other simulations. Interchange cannot currently produce PMEMD input files. Amber does not implement a switching function as [commonly used](https://openforcefield.github.io/standards/standards/smirnoff/#vdw) by SMIRNOFF force fields, so these force fields will produce different results in Amber than in OpenMM or GROMACS.

<!--
## CHARMM

An `Interchange` object can be written to CHARMM topology and
coordinate files with [`Interchange.to_psf()`] and [`Interchange.to_crd()`]:

```python
interchange.to_psf("out.to_psf")
interchange.to_crd("out.to_crd")
```
 -->
[`Interchange`]: openff.interchange.components.interchange.Interchange
[`Interchange.to_pdb()`]: openff.interchange.components.interchange.Interchange.to_pdb
[`Interchange.to_top()`]: openff.interchange.components.interchange.Interchange.to_top
[`Interchange.to_gro()`]: openff.interchange.components.interchange.Interchange.to_top
[`Interchange.to_mdp()`]: openff.interchange.components.interchange.Interchange.to_mdp
[`Interchange.to_gromacs()`]: openff.interchange.components.interchange.Interchange.to_gromacs
[`Interchange.to_lammps()`]: openff.interchange.components.interchange.Interchange.to_lammps
[`Interchange.to_openmm()`]: openff.interchange.components.interchange.Interchange.to_openmm
[`Interchange.to_prmtop()`]: openff.interchange.components.interchange.Interchange.to_prmtop
[`Interchange.to_inpcrd()`]: openff.interchange.components.interchange.Interchange.to_inpcrd
[`Interchange.to_sander_input()`]: openff.interchange.components.interchange.Interchange.to_sander_input
[`Topology.to_openmm()`]: openff.toolkit.topology.Topology.to_openmm
