# Advanced features

## Charge assignment logging

SMIRNOFF force fields support several different partial charge assignment methods. These are applied, in the following order

1. Look for preset charges from the `charge_from_molecules` argument
1. Look for chemical environment matches within the `<LibraryCharges>` section
1. Look for chemical environment matches within the `<ChargeIncrementModel>` section
1. Try to run AM1-BCC according to the `<ToolkitAM1BCC>` section or some variant 

If a molecule gets charges from one method, attempts to match charges for later methods are skipped. Note that preset charges override the force field and are not checked for consistency; any charges provided to the `charge_from_molecules` argument technically modify the force field. For more on how SMIRNOFF defines this behavior, see [this issue](https://github.com/openforcefield/standards/issues/68) and linked discussions.

Given this complexity, it may be useful to track how each atom actually got charges assigned. Interchange has opt-in logging to track this behavior. This uses the [standard library `logging` module](https://docs.python.org/3/library/logging.html) at the `INFO` level. The easiest way to get started is by adding something like `logging.basicConfig(level=logging.INFO)` to the beginning of a script or program. For example, this script:

```python
import logging

from openff.toolkit import ForceField, Molecule

logging.basicConfig(level=logging.INFO)

ForceField("openff-2.2.0.offxml").create_interchange(
    Molecule.from_smiles("CCO").to_topology()
)
```

will produce output including something like

```shell
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 0
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 1
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 2
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 3
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 4
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 5
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 6
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 7
INFO:openff.interchange.smirnoff._nonbonded:Charge section ToolkitAM1BCC, using charge method am1bcc, applied to (topology) atom index 8
```

This functionality is only available with SMIRNOFF force fields.
