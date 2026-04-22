# Advanced features

## Charge assignment logging

SMIRNOFF force fields support several different partial charge assignment methods. These are applied, in the following order

1. Look for preset charges from the `charge_from_molecules` argument
1. Look for chemical environment matches within the `<LibraryCharges>` section
1. Look for chemical environment matches within the `<ChargeIncrementModel>` section
1. Try to run AM1-BCC according to the `<ToolkitAM1BCC>` section or some variant

If a molecule gets charges from one method, attempts to match charges for later methods are skipped. Note that preset charges override the force field and are not checked for consistency; any charges provided to the `charge_from_molecules` argument technically modify the force field. For more on how SMIRNOFF defines this behavior, see [this issue](https://github.com/openforcefield/standards/issues/68) and linked discussions.

After all (mass-bearing) atoms have partial charges assigned, virtual sites are given charges by transferring charge from the atoms they are associated with ("orientation" atoms) according to the parameters of the force field. (The value of these parameters can be 0.0.)

Given this complexity, it may be useful to track how each atom actually got charges assigned. Interchange has opt-in logging to track this behavior. This uses the [standard library `logging` module](https://docs.python.org/3/library/logging.html) at the `logging.DEBUG` level. The easiest way to get started is by adding something like `logging.getLogger("openff.interchange").setLevel(logging.DEBUG)` to the beginning of a script or program. For example, this script:

```python
import logging

from openff.toolkit import ForceField, Molecule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openff.interchange").setLevel(logging.DEBUG)

ForceField("openff-2.2.0.offxml").create_interchange(
    Molecule.from_smiles("CCO").to_topology()
)
```

will produce output including something like

```shell
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 0
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 1
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 2
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 3
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 4
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 5
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 6
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 7
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bccelf10, applied to topology atom index 8
```

This functionality is only available with SMIRNOFF force fields.
