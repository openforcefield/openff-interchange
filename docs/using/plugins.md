# Creating custom interactions via plugins

Custom interactions (i.e. non-12-6 Lennard-Jones or non-harmonic bond or angle terms) can be introduced into the OpenFF stack via plugin interfaces. For each type of physical interaction in the model, plugins subclasses from `ParameterHandler` and `SMIRNOFFCollection` must be defined and exposed in the entry points `openff.toolkit.plugins.handlers` and `openff.interchange.plugins.collections`respectively.

## Creating custom `ParmaeterHandler`s

There is no formal specification for the `ParameterHandler`s as processed by the OpenFF toolkit, but there are some guidelines that are expected to be followed.

* The handler class _should_
  * inherit from `ParameterHandler`
  * contain a class that inherits from `ParameterType` (a "type class")
  * define a class attribute `_TAGNAME` that is used for serialization and handy lookups
  * define a class attribute `_INFOTYPE` that is the "type class"

* The handler class _may_
  * define a method `check_handler_compatibility` that is called when combining instances of this class
  * define other attributes for specific behavior, i.e. `scale14` for non-bonded interactions
  * override methods like `__init__` for highly custom behavior, but should be avoided if possible

* The "type class" _should_
  * define a class attribute `_ELEMENT_NAME` used for serialization
  * include attributes of each numerical value associated with a single parameter
    * Each of these attributes should be tagged with units and have default values (which can be `None`)

* The "type class" _may_
  * override methods like `__init__` for highly custom behavior, but should be avoided if possible
  * define its own `@property`s, class methods, etc. if needed
  * define optional attributes such as `id`

## Creating custom `SMIRNOFFCollection`s

## Examples

For example, here are two custom handlers. One defines a [Buckingham potential](https://en.wikipedia.org/wiki/Buckingham_potential) which can be used in place of a 12-6 Lennard-Jones potential.

```python
from openff.toolkit.typing.smirnoff.parameters import (
    ParameterHandler,
    ParameterType,
    ParameterAttribute,
    _allow_only,
    unit,
)


class BuckinghamHandler(ParameterHandler):
    class BuckinghamType(ParameterType):
        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c = ParameterAttribute(
            default=None,
            unit=unit.kilojoule_per_mole * unit.nanometer**6,
        )

    _TAGNAME = "Buckingham"
    _INFOTYPE = BuckinghamType

    scale12 = ParameterAttribute(default=0.0, converter=float)
    scale13 = ParameterAttribute(default=0.0, converter=float)
    scale14 = ParameterAttribute(default=0.5, converter=float)
    scale15 = ParameterAttribute(default=1.0, converter=float)

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="cutoff",
        converter=_allow_only(["cutoff", "PME"]),
    )

    combining_rules = ParameterAttribute(
        default="Lorentz-Berthelot",
        converter=_allow_only(["Lorentz-Berthelot"]),
    )
```

Notice that

* `BuckinghamHandler` (the "handler class") is a subclass of `ParameterHandler`
* `BuckinghamType` (the "type class")
  * is a subclass of `ParameterType`
  * defines `"Atom"` as its `_VALENCE_TYPE`, or chemical environment
  * defines `"Atom"` as its `_ELEMENT_TYPE`, which defines how it is serialized
  * has unit-tagged attributes `a`, `b`, and `c`, corresponding to particular values for each parameter
* the handler class also
  * defines a `_TAGNAME` for lookup and serialization
  * linkts itself with `BuckinghamType` via its `_INFOTYPE`
  * includes several optional attributes that are used in non-bonded interactions
    * `scale12` through `scale15`
    * the cutoff distance
    * the switch width
    * the method used to compute these interactions
    * the combining rules

From here we can instantiate this handler inside of a `ForceField` object, add some parameters (here, dummy values for water), and serialize it out to disk.

```python
from openff.toolkit import ForceField

handler = BuckinghamHandler(version="0.3")

handler.add_parameter(
    {
        "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
        "a": 1.0 * unit.kilojoule_per_mole,
        "b": 2.0 / unit.nanometer,
        "c": 3.0 * unit.kilojoule_per_mole * unit.nanometer**6,
    }
)


handler.add_parameter(
    {
        "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
        "a": 4.0 * unit.kilojoule_per_mole,
        "b": 4.0 / unit.nanometer,
        "c": 4.0 * unit.kilojoule_per_mole * unit.nanometer**6,
    }
)

force_field = ForceField()

force_field.register_parameter_handler(handler)

topology = Molecule.from_mapped_smiles("[H:2][O:1][H:3]").to_topology()

matches = force_field.label_molecules(topology)

# the matches of the 0th molecule in the handler tagged "Buckingham"
# printed as key-val pairs of atom indices and parameter types
print(*matches[0]["Buckingham"].items())
```

should output something like

```console
[((0,),
  <BuckinghamType with smirks: [#1]-[#8X2H2+0:1]-[#1]  a: 4.0 kilojoule_per_mole  b: 4.0 / nanometer  c: 4.0 kilojoule_per_mole * nanometer ** 6  >),
 ((1,),
  <BuckinghamType with smirks: [#1:1]-[#8X2H2+0]-[#1]  a: 1.0 kilojoule_per_mole  b: 2.0 / nanometer  c: 3.0 kilojoule_per_mole * nanometer ** 6  >),
 ((2,),
  <BuckinghamType with smirks: [#1:1]-[#8X2H2+0]-[#1]  a: 1.0 kilojoule_per_mole  b: 2.0 / nanometer  c: 3.0 kilojoule_per_mole * nanometer ** 6  >)]
```

This class should be registered as a plugin via the `entry_points` system by adding something like this to your `setup.py` or analogous setup file.

```python3
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "BuckinghamHandler = full.path.to.module:BuckinghamHandler",
        ],
```
