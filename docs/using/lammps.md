## Use with LAMMPS

An `Interchange` object can be writtent to a LAMMPS data file with `Interchange.to_lammps()`

```python3
interchange.to_lammps('data.lmp')
```

An input file with settings inferred from data in the `Interchange` object can also be written.
Note that this file will only run a single-point energy calculaqtion. `run` and other commands
should be modified to before running an MD simulation. This will write a file `run.inp`:

```python3
from openff.interchange.drivers.gromacs import _write_mdp_file
_write_lammps_input(interchange, "run.inp")
```
