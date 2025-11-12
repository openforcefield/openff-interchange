# Serialization

`Interchange` objects can be written to disk and read back at a later date (or on another machine, or by another person). Under the hood, serialization and de-serialization are largely handled by Pydantic, whose building block `pydantic.BaseModel` is a base class of `Interchange`. The serialization API is therefore inherited from Pydantic. It uses a mix of Python and Rust, so it's relatively fast.

There are a few use cases in which this can be useful, including but not limited to:

* You want to prepare system(s) on a laptop and run production simulations on a GPU cluster
* You want to prepare system(s) now and do something else with them at a later date
* You want to create a system once but make many modifications to it later (without re-running prep)
* You want to run triplicate (or more) simulations for statistics
* You want to run the same system in multiple engines and avoid run-to-run variance of a script

In some cases, writing engine-specific files to disk may serve a similar purpose. For GROMACS, Amber, and LAMMPS, use `Interchange`'s `to_gromacs`, `to_amber`, or `to_lammps` methods which generate files directly. With OpenMM, `Interchange.to_openmm` returns an `openmm.System` which itself can be serialized with [OpenMM's API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.XmlSerializer.html#openmm.openmm.XmlSerializer.serialize).

## Basic usage

```python
from openff.toolkit import ForceField, Molecule
from openff.interchange import Interchange

# prepare an Interchange object like normal
my_topology = Molecule.from_smiles("c1ccccc1").to_topology()
my_interchange = ForceField("openff-2.2.1.offxml").create_interchange(my_topology)

# see what the JSON string looks like - output will be large!
my_interchange.model_dump_json()

# or write directly to disk
open("benzene_interchange.json", "w").write(my_interchange.model_dump_json())

# read the same file back in
my_interchange2 = Interchange.model_validate_json(
    open("benzene_interchange.json").read()
)
```

## Caveats

1. Only JSON serialization is supported
1. On-disk representations are not stable _between_ versions before 1.0 (but stable within the same version)
1. No effort has been made, yet, to make JSON files small
