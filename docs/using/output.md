# Exporting to other software

## GROMACS

Once an [`Interchange`] object has been constructed, the `.gro` and `.top` files
can be written using [`Interchange.to_top()`] and [`Interchange.to_gro()`]:

```python
interchange.to_gro("out.gro")
interchange.to_top("out.top")
```

An `.mdp` file with settings inferred from data in the `Interchange` object can
also be written. Note that this file will only run a single-point energy
calculaqtion. `nsteps` and other lines should be modified to before running an
MD simulation. This will write a file `auto_generated.mdp`:

```python
from openff.interchange.drivers.gromacs import _write_mdp_file

_write_mdp_file(interchange)
```

## LAMMPS

An [`Interchange`] object can be written to a LAMMPS data file with
[`Interchange.to_lammps()`]

```python
interchange.to_lammps("data.lmp")
```

An input file with settings inferred from data in the `Interchange` object can
also be written. Note that this file will only run a single-point energy
calculation. `run` and other commands should be modified to before running an
MD simulation. This will write a file `run.inp`:

```python
from openff.interchange.drivers.lammps import _write_lammps_input

_write_lammps_input(interchange, "run.inp")
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

## Amber

Under construction!

## CHARMM

Under construction!

[`Interchange`]: openff.interchange.components.interchange.Interchange
[`Interchange.to_top()`]: openff.interchange.components.interchange.Interchange.to_top
[`Interchange.to_gro()`]: openff.interchange.components.interchange.Interchange.to_gro
[`Interchange.to_lammps()`]: openff.interchange.components.interchange.Interchange.to_lammps
[`Interchange.to_openmm()`]: openff.interchange.components.interchange.Interchange.to_openmm
