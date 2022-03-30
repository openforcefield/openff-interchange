# Exporting to other software

`Interchange` provides several methods to produce input data for other
software. Note that none of these methods write out all the information
stored in an `Interchange`; they support a design where the principle
source of truth is the rich chemical information in the `Interchange`
object, and exported files are tools to perform some operation.

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

<!--
:::{TODO}
We should either make this a public method or document it, not both
:::

An `.mdp` file with settings inferred from data in the `Interchange` object can
also be written. Note that this file will only run a single-point energy
calculation. `nsteps` and other lines should be modified to before running an
MD simulation. This will write a file `auto_generated.mdp`:

```python
from openff.interchange.drivers.gromacs import _write_mdp_file

_write_mdp_file(interchange)
``` -->

## LAMMPS

An [`Interchange`] object can be written to a LAMMPS data file with
[`Interchange.to_lammps()`]

```python
interchange.to_lammps("data.lmp")
```

<!--
:::{TODO}
We should either make this a public method or document it, not both
:::

An input file with settings inferred from data in the `Interchange` object can
also be written. Note that this file will only run a single-point energy
calculation. `run` and other commands should be modified to before running an
MD simulation. This will write a file `run.inp`:

```python
from openff.interchange.drivers.lammps import _write_lammps_input

_write_lammps_input(interchange, "run.inp")
``` -->

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

## Amber

An `Interchange` object can be written to Amber parameter/topology and
coordinate files with [`Interchange.to_prmtop()`] and [`Interchange.to_inpcrd()`]:

```python
interchange.to_prmtop("out.prmtop")
interchange.to_inpcrd("out.inpcrd")
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
