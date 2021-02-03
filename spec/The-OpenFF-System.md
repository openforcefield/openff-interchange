# The OpenFF System Specification

The OpenFF System provides an internal representation of parametrized systems for molecular simulation.

## Authors and acknowledgments

The OpenFF System specification was designed by the [Open Force Field Initiative](https://openforcefield.org).

Primary contributors include:
* Matt Thompson `<matt.thompson@openforcefield.org>`
* Jeff Wagner `<jeffrey.wagner@openforcefield.org>`

## Representation and encodings

The core of the OpenFF System is an internal (i.e., in computer memory) representation of chemical systems. This is in contrast to most models using in computational chemistry, which are based off of an on-disk representation.  Popular serialization formats accessible from Python dictionaries (i.e. XML, JSON and its derivatives) will be supported through a light serialization layer. Serialization to domain-specific formats (`.mol2`, `.pdb`, `.prmtop`, etc.) will be handled with a combination of native writers and conversion to external libraries.

## Primary components

Components will have discrete responsibilities in order to prevent scope creep and ambiguities about what data can be stored where. Wherever possible, components will have lists of "MUST, MAY, MUST NOT" that specify what data is necessary to construct or convert a component, what a component optionally may know, and what it is forbidden from knowing.

TODO: Populated tables like this for each component below

| MUST | MAY |  MUST NOT |
|---|---|---|
| Required things | Optional things | Forbidden things |

### Force field data

System parameters (force field parameters applied to a chemical topology) are represented as the sum of individual components (`PotentialHandler`s). Each term in a potential energy function is expected to be captured by a `PotentialHandler` or combination thereof. These closely mirror the `ParameterHandler`s in OpenFF Toolkit, and may merge in the future.

Each `PotentialHandler` subclass must specify an string-like `expression` that encodes the algebra of its energy evaluation and a collection of `independent_variables` that specify which variables in the expression do not need to be specified by system parameters. The remaining variables are then expected to be specified in a sequence of `Potential` objects stored in the handler.

```python3
>>> potential_handler = PotentialHandler(expression="m*x+b", independent_variables={"x"})
>>> potential_handler.expression
"m*x+b"
```

| MUST | MAY |  MUST NOT |
|---|---|---|
| Encode an energy evaluation method | Encode either an algebraic expression or tabluar data |
| Encode the necessary data to evaluate the functional form |
| Import from `PotentialHandler`

### Chemical topology

The [OpenFF Topology](https://open-forcefield-toolkit.readthedocs.io/en/0.7.2/api/generated/openforcefield.topology.Topology.html#openforcefield.topology.Topology) (or some future form of it) will serve as the representation of the chemical topology. This encodes a graph-like representation of the system _but no physics_. The envisioned scope of this component is roughly equivalent to the data that can be stored in an SDF file.

| MUST | MAY |  MUST NOT |
|---|---|---|
| Include particles with masses | Include atoms with elements
| | Include virtual or coarse-grained particles
| Include all existing bonds | Include constraints, fractional bond orders,
| | Include arbitrary metadata (cheminformatics tags, residue/chain/segments)
| | | Specify periodicity or box vectors

### Positions

Positions of all particles are stored in a unit-bearing Nx3 matrix. Here, particles refers to all atoms and virtual sites in the system.

| MUST | MAY |  MUST NOT |
|---|---|---|
| Include an array of particle positions | Include virtual particles
| Have shape `(N, 3)` where `N` is the total number of particles (atoms and virtual particles) in the topology
| Tag positions with any unit of length dimensionality | Include positions without units (which will be tagged with the units of the object model)

Third-party libraries, i.e. [`mBuild`](https://mbuild.mosdef.org/en/stable/) or [`PACKMOL`](http://m3g.iqm.unicamp.br/packmol/home.shtml), can be used to  can be used to modify the positions of particles. A modified array can be re-attched to a system using a setter, which handles input validation.

### Box vectors

Box vectors are represented as a 3x3 matrix of unit-bearing quantities that specify the periodicity of the dimensions of a periodic simulation cell. Both orthogonal and triclinic boxes are supported. For details, see the [OpenMM implementation](http://docs.openmm.org/latest/userguide/theory.html#periodic-boundary-conditions).

Box vectors are optional. In order to represent non-periodic (i.e. vacuum) systems, `None` is an valid value.

| MUST | MAY |  MUST NOT |
|---|---|---|
| Be a 3x3 array of atomic positions | Be `None` to encode non-periodicity | Positions with implicit or ambiguous units
| Tag vectors with any unit of length dimensionality | Include vectors without units (which will be tagged with the units of the object model)

## Interoperability

This specification does not assume first-class compatibility with any particular molecular simulation engines or software stacks. It instead aims to provide a general and flexible model for storing data and separate out interoperability tasks as a separate layer.

Separating the internal representation from the interoperability layer allows for a clear layer between them in which sanity checks can be carried out and compand compatibility checks can be encoded. For example, if an object in memory includes CMAP terms, it cannot be written to Cassandra, which does not support CMAPs, or if an object in memory includes Buckingham potentials, it cannot be converted to a ParmEd Structure.

## Features

### Tracking parameter sources

Given a parametrized system, it is typically difficult and often impossible to track the origin of parameter. Here, all potential handlers will expose methods that point to their source parameters. For each parameter in a parametrized system, there must be a method that uniquely indefinies its source of the data, i.e. a unique identifier in a force field and the identity of that force field.

### Exposing differential representations

At a fundamental level, a parametrized system is just a specialized view into a set of numbers. To latch onto existing infrastructure in other communities, particularl machine learning, potential handlers will include methods that expose their data via array representations that can be autodiffed.

### System combination

Systems can be combined with the built-in `+` operator. Internal checks will be done to ensure appropriate compatibility of components. There can also be a high-level function (like ) There can also be a high-level function (like `.add_component`) that serves as a broad API point for several more specific functions.

### Energy evaluation

A high-level `.get_energy()` function will be exposed with a switchable backend. This enables the use of different engines (i.e. an OpenMM driver, a GROMACS driver, etc.) and may include an internal driver.

## Advanced features

### SMIRNOFF implementation

The OpenFF System will implement all features of the [SMIRNOFF](https://open-forcefield-toolkit.readthedocs.io/en/0.7.2/smirnoff.html) specification.

### Extensibility to new potentials

The base class for potential handlers specifies the minimum amount of information required to write a new handler. If a user wishes to write a custom handler, they can either subclass out of this base class or instantiate it directly with a custom set of data and parametrization rules. For example, a handler encoding the Buckingham potential could be created mostly by copying a Lennard-Jones non-bonded handler and modifying the `expression` attribute (assuming that the parametrization machinery used for the Lennard-Jones-based terms can safely drop in).  Handlers storing more novel terms, such as CMAPs, will require more custom code but will fit into the framework of the base class.

### Partial re-parametrization

Some handlers may encode special triggers for re-parametrization. For example, consider a `ProperTorsionHandler` that relies on bond order interpolation. This typing can depends on a fractional bond order assignment method. If that is modified, the parametrization (of only one handler) could be re-triggered.

### Electrostatics meta-handler

Potential handlers are meant to be highly modular with lifecycles that act independently of all others, i.e. the entire parametrization process, data structures, and utilities of the bonds handler exist independently of the angle handler.  Electrostatics cannot behave this way, as the final parametrization (a set of partial charges) is often the sum of smaller handlers (a combination of library charges, partial charge assignment methods, charge increments, etc.) These handlers, not only the partial charges, need to be accessible to optimization routines and user inspection.

### Batch modification of parameters

Some amount (all?) of parameters will be exposed to modification by the user.

Say, for example, one wanted to scale all partial charges in a system by a constant factor.

```python3
factor = 0.8
electrostatics = system.get_handler('Electrostatics')
for potential in electrostatics.potentials:
    potential.charge *= factor
```

Each handler will need to define its own rules for what modifications are allowed and, after modification, what status check(s) must pass.

### User control over system combination

A plug-in architecture will expose settings for the level of rigour executed in the internal consistency checks. The default will be strict, resulting in errors if there is any ambiguity in the physical description of each system (i.e. different non-bonded cutoff treatments). A small number (1-2) of other settings with more granularity will be exposed, i.e. one that is problematively permissive and another that implements some amount of "reasonable" discrepancies to be fudged together. This will all be constructed in a way that enables users to define their own sets of "knobs" for when system combination can and cannot be allowed, i.e. if aromaticity models need not match but cut-off treatments must.

## Usage examples

### Creating a system from scratch

### Piecing together a system from scattered components

### Adding systems together

### Partial modification and re-parametrization

## Implementation details

### Pydantic BaseModel

The core components are built off of the `BaseModel` class of [Pydantic](https://pydantic-docs.helpmanual.io/), which provides a framework for rigorous type validation and serialization pathways through dictionaries.

### Python API

The reference implementation is the [openff-system](https://github.com/openforcefield/openff-system) Python package, based off of the MolSSI [CMS cookiecutter](https://github.com/molssi/cookiecutter-cms). It includes a Pythonic API, automated unit and integration tests, versioning, and continuous integration, and extensive linting of the source code. It is developed and maintained by the Open Force Field Initiative and released under the MIT license.

### Representation of physical quantities

Quantities (both array and scalar data) stored internally must be tagged with units via `[Pint](https://github.com/hgrecco/pint)` or a close derivative. Some interfaces may expose unitless quantities, if necessary, but most represnetations exposed to the user must be unit-tagged. Implementation details, particularly relating to compatibility with the rest of the OpenFF stack, are to be determined.

## Relevant edge cases

* Conception of "pre-" and "post-typing" topologies
* Do virtual sites go in the topology, a special "post-typing" topology, or should they be computed on-the-fly as needed?
* Allowing/forbidding/tracking/fixing "dirty" states
* Dealing with mal-formed files or those that play fast-and-loose with specifications (MOL2, PDB, etc.)
* Safely supporting alchemical mutations
* Tracking alchemical mutations (safely storing a diff?)
* How to handle polarizability
* Tracking rigid bodies/freeze groups
* Supporting HMR or other isotropic-like mutations
* Tracking the _history_ of a parameter in addition to only its source
* Storing `expression`-like data for mathematically complex potentials like machine-learning based models

## Version history

### 0.0

Initial draft specification

### References

* https://github.com/openforcefield/openforcefield/issues/310
*
