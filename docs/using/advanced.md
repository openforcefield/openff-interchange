# Advanced features

## Charge assignment logging

SMIRNOFF force fields support several different partial charge assignment methods and [employ a hierarchical scheme](smirnoff-charge-assignment-hierarchy) to determine which is used on each molecule. Given this complexity, it may be useful to track how each molecule's atoms actually had their charges assigned. (Note that, except for some complexity with `<ChargeIncrementModel>`, all atoms in a given molecule are assigned charges via the same method when using SMIRNOFF force fields.) Interchange has opt-in logging to track this behavior. This uses the [standard library `logging` module](https://docs.python.org/3/library/logging.html) at the `DEBUG` level. The easiest way to get started is by adding something like `logging.getLogger("openff.interchange").setLevel(logging.DEBUG)` to the beginning of a script or program. For example, this script:

```python
import logging

from openff.toolkit import ForceField, Molecule, Topology

logging.basicConfig()
logging.getLogger("openff.interchange").setLevel(logging.DEBUG)

ForceField("openff-2.3.0.offxml").create_interchange(
    Topology.from_molecules(
        [
            Molecule.from_smiles("CCO"),
            Molecule.from_smiles("O"),
            Molecule.from_smiles("O"),
            Molecule.from_smiles(341 * "C"),
        ],
    )
)
```

will produce output including something like

```shell
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section NAGLCharges, using NAGL model openff-gnn-am1bcc-1.0.0.pt, applied to molecule with Hill formula C2H6O and InChIKey LFQSCWFLJHTTHZ-UHFFFAOYSA-N
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section LibraryCharges, applied to molecule with Hill formula H2O and InChIKey XLYOFNOQVPJJNP-UHFFFAOYSA-N
DEBUG:openff.interchange.smirnoff._nonbonded:Charge section NAGLCharges, using NAGL model openff-gnn-am1bcc-1.0.0.pt, applied to molecule with Hill formula C341H684 and InChIKey CLLUCSPVKWTWLW-UHFFFAOYSA-N
```

This functionality is only available with SMIRNOFF force fields.
