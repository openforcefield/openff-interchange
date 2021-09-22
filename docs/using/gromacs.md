# Use with GROMACS

Once an `Interchange` object has been constructed, the `.gro` and `.top` files can be written using
`Interchange.to_top` and `Interchange.to_gro`:

```python3
interchange.to_gro("out.gro")
interchange.to_top("out.top")
```

An `.mdp` file with settings inferred from data in the `Interchange` object can also be written.
Note that this file will only run a single-point energy calculaqtion. `nsteps` and other lines
should be modified to before running an MD simulation. This will write a file `auto_generated.mdp`:

```python3
from openff.interchange.drivers.gromacs import _write_mdp_file
_write_mdp_file(interchange)
```
