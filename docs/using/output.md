# Exporting to other software

`Interchange` provides several methods to produce input data for other
software. Note that none of these methods write out all the information
stored in an `Interchange`; they support a design where the principle
source of truth is the rich chemical information in the `Interchange`
object, and exported files are tools to perform some operation.

(sec-mdconfig)=
## Run control/config files

SMIRNOFF force fields include several parameters that many MD engines do not
include as part of their topologies. These values are essential for accurately
simulating output from Interchange, but they are configured in the same files
that are used for general control of simulation runtime behavior. As a result,
Interchange cannot simply provide complete versions of these files.

Instead, Interchange provides [`MDConfig`], a class that writes stub versions of
MD engine run input files. These files must be modified and completed before
they can be used to run a simulation.

`MDConfig` can be constructed from an existing Interchange:

```python
from openff.interchange import Interchange
from openff.interchange.components.mdconfig import MDConfig

interchange = Interchange.from_smirnoff(...)

mdconfig = MDConfig.from_interchange(interchange)
```

[`MDConfig`]: openff.interchange.components.mdconfig.MDConfig

## General purpose

An [`Interchange`] can be written out as the common PDB structure format
with the [`Interchange.to_pdb()`] method:

```python
interchange.to_pdb("out.pdb")
```

## GROMACS

Once an [`Interchange`] object has been constructed, the `.gro` and `.top` files
can be written using [`Interchange.to_top()`] and [`Interchange.to_gro()`]:

```python
interchange.to_gro("out.gro")
interchange.to_top("out.top")
```

A .MDP file can be written from an [MDConfig object] constructed from the
interchange. The resulting file will run a single-point energy calculation and
should be modified for the desired simulation:

```python
mdconfig.write_mdp_file(mdp_file="auto_generated.mdp")
```

## LAMMPS

An [`Interchange`] object can be written to a LAMMPS data file with
[`Interchange.to_lammps()`]

```python
interchange.to_lammps("data.lmp")
```

An input file can be written from an [MDConfig object] constructed from the interchange. The resulting file will run a single-point energy calculation and
should be modified for the desired simulation:

```python
mdconfig.write_lammps_input(input_file="auto_generated.in")
```

## OpenMM

An [`Interchange`] object can be converted to an `openmm.System` object with
[`Interchange.to_openmm()`].

```python
openmm_sys = interchange.to_openmm()
```

By default, this will separate non-bonded interactions into several different
`openmm.Force` objects. To combine everything into a single
`openmm.NonbondedForce`, use the `combine_nonbonded_forces=True` argument.

The accompanying OpenMM topology can be constructed with the
[`Topology.to_openmm()`] method:

```python
openmm_top = interchange.topology.to_openmm()
```

Recall that all unit-bearing attributes within `Interchange` objects are `openff.units.Quantity` objects, which can be converted out to `openmm.unit.Quantity` objects via their `.to_openmm()` method. For example:

```python
openmm_positions: openmm.unit.Quantity = interchange.positions.to_openmm()
openmm_box: openmm.unit.Quantity = interchange.box.to_openmm()
```

## Amber

An `Interchange` object can be written to Amber parameter/topology and
coordinate files with [`Interchange.to_prmtop()`] and [`Interchange.to_inpcrd()`]:

```python
interchange.to_prmtop("out.prmtop")
interchange.to_inpcrd("out.inpcrd")
```

A run control file can be written from an [MDConfig object] constructed from the
interchange. The resulting file will run a single-point energy calculation and
should be modified for the desired simulation:

```python
mdconfig.write_sander_input_file(input_file="auto_generated.in")
```

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
[`Interchange.to_gro()`]: openff.interchange.components.interchange.Interchange.to_gro
[`Interchange.to_lammps()`]: openff.interchange.components.interchange.Interchange.to_lammps
[`Interchange.to_openmm()`]: openff.interchange.components.interchange.Interchange.to_openmm
[`Interchange.to_prmtop()`]: openff.interchange.components.interchange.Interchange.to_prmtop
[`Interchange.to_inpcrd()`]: openff.interchange.components.interchange.Interchange.to_inpcrd
[`Interchange.to_psf()`]: openff.interchange.components.interchange.Interchange.to_psf
[`Interchange.to_crd()`]: openff.interchange.components.interchange.Interchange.to_crd
[`Topology.to_openmm()`]: openff.toolkit.topology.Topology.to_openmm
[MDConfig object]: sec-mdconfig
