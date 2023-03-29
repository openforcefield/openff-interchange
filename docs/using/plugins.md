# Creating custom interactions via plugins

Custom interactions (i.e., interactions other than 12-6 Lennard-Jones, harmonic bonds, or harmonic angles) can be introduced into the OpenFF stack via plugin interfaces. To create a plugin, create a subclass of both [`ParameterHandler`] and [`SMIRNOFFCollection`] for each new type of physical interaction in the model, then expose them to Interchange via the `openff.toolkit.plugins.handlers` and `openff.interchange.plugins.collections` [SetupTools entry points], respectively.

Parameter handlers are responsible for the system- and engine-independent aspects of parametrization; in other words, they define the force field itself. By contrast, a SMIRNOFF collection handles the parametrization of a particular system and the preparation of that system for simulation with an external MD engine. The base classes for parameter handlers are defined in the OpenFF Toolkit, while collections are defined in Interchange.

Currently, systems using custom interactions can only be exported to OpenMM. There are two routes for this export: one using existing Interchange machinery and another that allows for completely custom behavior.

[`ParameterHandler`]: openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler
[`SMIRNOFFCollection`]: openff.interchange.plugins.SMIRNOFFCollection
[SetupTools entry points]: setuptools:userguide/entry_point
[`openmm.Force`]: openmm.openmm.Force

## Creating a custom `ParameterHandler`

There is no formal specification for a `ParameterHandler` as processed by the OpenFF Toolkit, but there are some guidelines that are expected to be followed. A parameter handler requires both a [`ParameterHandler`] class, which performs parametrization, and a [`ParameterType`] class, which stores the corresponding data. For more information, see the [Toolkit documentation](users/developing.md#modular-design-features).

The high-level objectives of a parameter handler are to:

* store force field parameters
* assign parameters via SMIRKS
* enable construction and modification of parameters via the Python API
* enable serialization and deserialization of parameters

* The handler class _should_
  * inherit from [`ParameterHandler`]
  * be available in the `openff.toolkit.plugins.handlers` [entry point group]
  * contain a class that inherits from [`ParameterType`] (a "type class")
  * define a class attribute `_TAGNAME: str` that is used for serialization and handy lookups
  * define a class attribute `_INFOTYPE` that specifies the handler's "type class"

* The handler class _may_
  * define a method [`check_handler_compatibility`] that is called when combining instances of this class
  * define other attributes for specific behavior, i.e. `scale14` for non-bonded interactions
  * override methods like `__init__` for custom behavior if required (this should be avoided if possible)

* The "type class" _should_
  * inherit from [`ParameterType`]
  * define a class attribute `_ELEMENT_NAME: str` used for serialization
  * include attributes defining each numerical value associated with a single parameter
    * Each of these attributes should be tagged with units and have default values (which can be `None`)

* The "type class" _may_
  * override methods like `__init__` for custom behavior if required (this should be avoided if possible)
  * define its own properties, class methods, etc. as needed
  * define optional attributes such as `id`

[`ParameterType`]: openff.toolkit.typing.engines.smirnoff.ParameterType
[entry point group]: setuptools:userguide/entry_point
[`check_handler_compatibility`]: openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler.check_handler_compatibility

## Creating a custom `SMIRNOFFCollection`

There is similarly no specification for these plugins yet, but some guidelines that should be followed.

A `SMIRNOFFCollection` is a subclass of [`Collection`] specific to SMIRNOFF force fields. The objectives of a collection are to:

* store a system's parameters, as assigned by SMIRKS-based typing of chemical topologies
* provide an interface for exporting these parameters to MD engines (i.e. creating an [`openmm.Force`])

* The class _should_
  * inherit from [`SMIRNOFFCollection`] and therefore be written in the style of a [Pydantic model].
  * define a field [`is_plugin: bool = True`]
  * be available in the `openff.interchange.plugins.collections` [entry point group]
  * define a field [`type: str`], which we suggest match the corresponding `ParameterHandler._TAGNAME`
  * define a field [`expression: str`], a mathematical formula defining how it contributes to the overall potential energy
  * define a class method [`allowed_parameter_handlers`] that returns an iterable of the `ParameterHandler` subclasses that it can process
  * define a class method [`supported_parameters`] that returns an iterable of parameters that it expects to store (i.e. `"smirks"`, `"k"`, `"length"`, etc.)
  * override other methods of [`SMIRNOFFCollection`] and  [`Collection`] ([`store_matches`], [`store_potentials`], [`create`]) as needed

* The class _may_
  * define other optional fields, similar to optional attributes on its corresponding parameter handler
  * define other methods and fields as needed

[Pydantic model]: https://docs.pydantic.dev/usage/models/
[`is_plugin: bool = True`]:openff.interchange.plugins.SMIRNOFFCollection.is_plugin
[`type: str`]: openff.interchange.components.potentials.Collection.type
[`expression: str`]:openff.interchange.components.potentials.Collection.expression
[`allowed_parameter_handlers`]:openff.interchange.plugins.SMIRNOFFCollection.allowed_parameter_handlers
[`supported_parameters`]:openff.interchange.plugins.SMIRNOFFCollection.supported_parameters
[`Collection`]: openff.interchange.components.potentials.Collection
[`store_matches`]: openff.interchange.plugins.SMIRNOFFCollection.store_matches
[`store_potentials`]: openff.interchange.components.potentials.Collection.store_potentials
[`create`]: openff.interchange.plugins.SMIRNOFFCollection.create

## Examples

### Bootstrapping existing Interchange machinery

For example, here are two custom handlers. One defines a [Buckingham potential](https://en.wikipedia.org/wiki/Buckingham_potential) which can be used in place of a 12-6 Lennard-Jones potential.

```python
from openff.toolkit.typing.engines.smirnoff.parameters import (
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
    * the combining rules (which will not be used with water)

From here we can instantiate this handler inside of a [`ForceField`] object, add some parameters (here, dummy values for water), and serialize it out to disk:

```python
from openff.toolkit import ForceField, Molecule

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

force_field = ForceField(load_plugins=True)

force_field.register_parameter_handler(handler)

topology = Molecule.from_mapped_smiles("[H:2][O:1][H:3]").to_topology()

matches = force_field.label_molecules(topology)

# the matches of the 0th molecule in the handler tagged "Buckingham"
# printed as key-val pairs of atom indices and parameter types
print(*matches[0]["Buckingham"].items())
```

[`ForceField`]: openff.toolkit.typing.engines.smirnoff.ForceField

Which should output something like:

```console
[((0,),
  <BuckinghamType with smirks: [#1]-[#8X2H2+0:1]-[#1]  a: 4.0 kilojoule_per_mole  b: 4.0 / nanometer  c: 4.0 kilojoule_per_mole * nanometer ** 6  >),
 ((1,),
  <BuckinghamType with smirks: [#1:1]-[#8X2H2+0]-[#1]  a: 1.0 kilojoule_per_mole  b: 2.0 / nanometer  c: 3.0 kilojoule_per_mole * nanometer ** 6  >),
 ((2,),
  <BuckinghamType with smirks: [#1:1]-[#8X2H2+0]-[#1]  a: 1.0 kilojoule_per_mole  b: 2.0 / nanometer  c: 3.0 kilojoule_per_mole * nanometer ** 6  >)]
```

This class should be registered as a plugin via the `entry_points` system by adding something like this to your `setup.py` or analogous setup file.

```python
entry_points = {
    "openff.toolkit.plugins.handlers": [
        "BuckinghamHandler = full.path.to.module:BuckinghamHandler",
    ],
}
```

At this point, we have created a class that can parse sections of a custom OFFXML file (or create this handler from the Python API). We next need to create a custom `SMIRNOFFCollection` to process these parameters. In this case, we are using a shortcut by inheriting from `_SMIRNOFFNonbondedCollection`, which itself inherits from `SMIRNOFFCollection` and adds in some default fields for non-bonded interactions.

```python
from openff.toolkit import Topology

from typing import Literal, Type
from openff.models.types import FloatQuantity
from openff.interchange.smirnoff._nonbonded import _SMIRNOFFNonbondedCollection
from openff.interchange.components.potentials import Potential


class SMIRNOFFBuckinghamCollection(_SMIRNOFFNonbondedCollection):
    type: Literal["Buckingham"] = "Buckingham"

    expression: str = "a*exp(-b*r)-c/r**6"

    method: str = "cutoff"

    mixing_rule: str = "Buckingham"

    switch_width: FloatQuantity["angstrom"] = unit.Quantity(1.0, unit.angstrom)

    @classmethod
    def allowed_parameter_handlers(cls):
        return [BuckinghamHandler]

    @classmethod
    def supported_parameters(cls):
        return ["smirks", "id", "a", "b", "c"]

    def store_potentials(self, parameter_handler: BuckinghamHandler) -> None:
        self.method = parameter_handler.method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            self.potentials[potential_key] = Potential(
                parameters={
                    "a": parameter.a,
                    "b": parameter.b,
                    "c": parameter.c,
                },
            )

    @classmethod
    def create(
        cls,
        parameter_handler: BuckinghamHandler,
        topology: Topology,
    ):
        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            method=parameter_handler.method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler
```

One last piece of housekeeping: because we are defining these plugins as one-off classes in an interactive session, some steps in the plugin loading machinery were skipped over. To get this to work without creating a separate module, we need to monkey-patch two private classes; one that tracks which SMIRNOFF sections are supported, and another that maps handlers to collection classes:

```python
from openff.interchange.smirnoff._create import (
    _SUPPORTED_PARAMETER_HANDLERS,
    _PLUGIN_CLASS_MAPPING,
)


_SUPPORTED_PARAMETER_HANDLERS.add("Buckingham")
_PLUGIN_CLASS_MAPPING[BuckinghamHandler] = SMIRNOFFBuckinghamCollection
```

With everything registered and still in memory, we can create an `Interchange` using it and a simple water [`Topology`]:

```python
from openff.interchange import Interchange

Interchange.from_smirnoff(force_field, topology)
```

[`Topology`]: openff.toolkit.topology.Topology

### Creating completely custom behavior

For cases in which custom behavior has no analog in existing Interchange functionality, another route is available. Define on the collection the [`modify_openmm_forces`] method as follows:

```python
class MySMIRNOFFCollection(SMIRNOFFCollection):
    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        ...
```

[`modify_openmm_forces`]: openff.interchange.plugins.SMIRNOFFCollection.modify_openmm_forces

This provides complete access to the contents of the [`Interchange`] object and the flexibility to modify the [`openmm.System`] as desired - modifying, adding, deleting, etc. existing particles, forces, exclusions, etc. This method is internally called on each plugin collection when calling [`Interchange.to_openmm()`] via a code block like:

```python
def to_openmm(
    interchange,
    combine_nonbonded_forces: bool = False,
    add_constrained_forces: bool = False,
) -> openmm.System:
    ...
    # Process in-spec SMIRNOFF sections
    ...

    for collection in interchange.collections.values():
        if collection.is_plugin:
            try:
                collection.modify_openmm_forces(
                    interchange,
                    system,
                    add_constrained_forces=add_constrained_forces,
                    constrained_pairs=constrained_pairs,
                    particle_map=particle_map,
                )
            except NotImplementedError:
                continue

    return system
```

[`Interchange`]: openff.interchange.Interchange
[`openmm.System`]: openmm.openmm.System
[`Interchange.to_openmm()`]: openff.interchange.Interchange.to_openmm()
