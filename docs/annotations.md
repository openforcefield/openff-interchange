# Annotations

This document was written in November 2024 assuming

* Python 3.10+
* Pydantic V2 (not the V1 API backdoor)
* Mypy 1.10+ or similarly-functional Python type-checker
* Interchange 0.4+

Consider a class to store information about atoms. It needs to store an atomic mass, a partial charge, and a position in 3-D space. The following class would work fine:

```python
class Atom:
    mass: float
    charge: float
    position: list
```

but this has some shortcomings:

* What units are assumed for each attribute?
* What if the position is a `numpy.ndarray` (or `openmm.Vec3`)?
* Can instances of this class be serialized to disk? And then de-serialized?
* Are any fields optional?

OpenFF's combined use of [Pint](https://pint.readthedocs.io/en/stable/) and [Pydantic](https://docs.pydantic.dev/latest/) solves these problems. To leverage these tools,

1. Classes should subclass out of `openff.interchange.pydantic._BaseModel`, which is a thin wrapper around `pydantic.BaseModel`
2. Annotations should use `openff.interchange._annotations._Quantity`, which is a thin wrapper around `openff.units.Quantity`, itself having the same behavior as `pint.Quantity`, or annotations derived thereof.

The same class can be rewritten:

```python
from openff.interchange.pydantic import _BaseModel
from openff.interchange._annotations import _Quantity
from openff.units import Quantity


class Atom(_BaseModel):
    mass: _Quantity
    charge: _Quantity
    position: _Quantity | None = None


carbon = Atom(
    mass=_Quantity("12.011 amu"),
    charge=_Quantity("-0.1 elementary_charge"),
    position=_Quantity([0, 0, 0], "nanometer"),
)

print(carbon.model_dump())
# {'mass': {'val': 12.011, 'unit': 'unified_atomic_mass_unit'}, 'charge': {'val': -0.1, 'unit': 'elementary_charge'}, 'position': {'val': [0, 0, 0], 'unit': 'nanometer'}}

print(carbon.model_dump_json())
# {"mass":{"val":12.011,"unit":"unified_atomic_mass_unit"},"charge":{"val":-0.1,"unit":"elementary_charge"},"position":{"val":[0,0,0],"unit":"nanometer"}}

print(Atom.model_validate_json(carbon.model_dump_json()))
# Atom(mass=<Quantity(12.011, 'unified_atomic_mass_unit')>, charge=<Quantity(-0.1, 'elementary_charge')>, position=<Quantity([0 0 0], 'nanometer')>)
```

This is an improvement, but units are still implicit. These annotations can be more specific, optionally defining the dimensionality or the _specific_ unit. The `_annotations` module defines a few of these out of the box for convenience, but more can be defined by the user using the `typing.Annotated` pattern in the source code.

```python
from openff.interchange._annotations import _ElementaryChargeQuantity, _DistanceQuantity


class Atom(_BaseModel):
    mass: _Quantity
    charge: _ElementaryChargeQuantity
    position: _DistanceQuantity | None = None
```

With the use of `_DistanceQuantity`, there is a dimensionality check during validation that prevents incompatible units from being passed (compatible units will checked, but not converted):

```python
try:
    carbon = Atom(
        mass=Quantity("12.011 amu"),
        charge=Quantity("-0.1 elementary_charge"),
        position=Quantity([1, 2, 3], "kJ/mol"),
    )
except ValidationError as error:
    print(str(error))
# 1 validation error for Atom
# position
#   Value error, Dimensionality of quantity=<Quantity([1 2 3], 'kilojoule / mole')> is not compatible with unit='nanometer' [type=value_error, input_value=<Quantity([1 2 3], 'kilojoule / mole')>, input_type=Quantity]
#     For further information visit https://errors.pydantic.dev/2.9/v/value_error
```

The charage field was defined with an annotation that specifies not only dumensionality, but the _specific unit_. In this case, inputs are validated and converted to the specified units:

```python
print(
    Atom(
        mass=Quantity(12.011, "amu"),
        charge=Quantity("-0.1 / 6.241509074460763e+18 coulomb"),
    ).charge
)
# -0.10000000000000002 elementary_charge
```

This can be useful if i.e. it is important that positions are universally stored with Angstroms and not nanometers.
