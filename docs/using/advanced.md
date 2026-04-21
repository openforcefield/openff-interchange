# Advanced features

## Charge assignment logging

SMIRNOFF force fields support several different partial charge assignment methods and [employ a hierarchical scheme](smirnoff-charge-assignment-hierarchy) to determine which is used on each molecule. Given this complexity, it may be useful to track how each molecule's atoms actually had their charges assigned. (Note that, except for some complexity with `<ChargeIncrementModel>`, all atoms in a given molecule are assigned charges via the same method when using SMIRNOFF force fields.) Interchange has opt-in logging to track this behavior. This uses the [standard library `logging` module](https://docs.python.org/3/library/logging.html) at the `INFO` level. The easiest way to get started is by adding something like `logging.basicConfig(level=logging.INFO)` to the beginning of a script or program. For example, this script:

```python
import logging

from openff.toolkit import ForceField, Molecule, Topology

logging.basicConfig(level=logging.INFO)

ForceField("openff-2.2.0.offxml").create_interchange(
    Topology.from_molecules(
        [
            Molecule.from_smiles("CCO"),
            Molecule.from_smiles("O"),
        ],
    )
)
```

will produce output including something like

```shell
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to molecule with InChI InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3
INFO:openff.interchange.smirnoff._nonbonded:Charge section LibraryCharges, applied to molecule with InChI InChI=1S/H2O/h1H2
```

This functionality is only available with SMIRNOFF force fields.
