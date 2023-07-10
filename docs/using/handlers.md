# Tweaking and Inspecting Parameters

Interchange stores force field parameters in potential handlers, which link the
entry in the force field to the parameter as applied to a topology. This makes
it easy to inspect and even modify how a parameter is applied to a system.

## Collections

Collections describe how force field parameters are applied to a chemical
system. This attribute is a dictionary whose keys are string identifiers and
whose values are [`Collection`] objects. Instead of linking parameters to
abstract chemical environments like SMIRNOFF force fields, or copying
parameters into place like a traditional force field format, a
`Collection` links an input parameter in the force field to every place
in the topology where it is used.

Like SMIRNOFF force fields, each parameter in an `Interchange`  knows where it
came from, but like traditional force fields, the parameterized system is
readily defined and values can be retrieved quickly. This allows changes to an
input parameter to be reflected instantly in the parameterized system. Unlike a
SMIRNOFF force field, the chemistry of system itself cannot be changed; a new
`Interchange` must be defined and parameterized.

There are three central components in each handler: topology keys, potentials,
and potential keys.

[`TopologyKey`] objects are unique identifiers of locations in a topology. These
objects do not include physics parameters. The basic information is a tuple of
atom indices, which can be of any non-zero length. For example, a topology key
describing a torsion will have a 4-length tuple, and a topology key describing
a vdW parameter will have a 1-length tuple.

[`Potential`] objects store the physics parameters that result from parameterizing
a chemical topology with a force field. These do not know anything about where
in the topology they are applied. The parameters are stored in a dictionary
attribute `.parameters` in which keys are string identifiers and values are the
parameters themselves, tagged with units.

[`PotentialKey`] objects uniquely identify physics parameters so that many
topology keys can refer to the same potential. Potential keys do not know
anything about the topology they are associated with. In SMIRNOFF force fields,
SMIRKS patterns uniquely identify a parameter within a parameter handler, so
(with some exceptions) the SMIRKS pattern is all that is needed to construct a
potential key. For classically atom-typed force fields, a key can be
constructed using atom types or combinations thereof.

These objects are strung together with two mappings, each stored as dictionary
attributes of a `Collection`. The `.key_map` attribute maps segments of
a topology to the potential keys (`TopologyKey` to `PotentialKey` mapping). The
`.potentials` attribute maps the potential keys to the potentials
(`PotentialKey` to `Potential`). This allows many topology keys to map to the
same `Potential` by sharing a `PotentialKey`. If the `Potential` is updated,
all the places in the topology where it is used are updated immediately.
Despite this, getting the `Potential` for a place in the topology is a constant
time operation. For example, parametrizing a thousand water molecules each with
two identical bonds will produce only one `Potential`, rather than two thousand.

Each potential handler inherits from the base [`Collection`] class and
describes a single type of parameter from a single source. Potential handlers
for SMIRNOFF force fields are found in the [`openff.interchange.smirnoff`]
module, while those for Foyer are found in the [`openff.interchange.foyer`]
module.

## Inspecting an assigned parameter

Construct a simple `Interchange`

```pycon
>>> from openff.interchange import Interchange
>>> from openff.toolkit import Molecule, ForceField
>>>
>>> ethane = Molecule.from_smiles("CC")
>>> sage = ForceField("openff-2.0.0.offxml")
>>> box = box = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
>>> interchange = Interchange.from_smirnoff(sage, [ethane], box=box)

```

The [`Interchange.handlers`] attribute maps names to the corresponding handler:

```pycon
>>> interchange.handlers.keys()  # doctest: +NORMALIZE_WHITESPACE,+ELLIPSIS
dict_keys(['Bonds', 'Constraints', 'Angles', 'ProperTorsions',
    'ImproperTorsions', 'vdW', 'Electrostatics'])
>>> # Ethane has no improper torsions, so both maps will be empty
>>> interchange.handlers['ImproperTorsions']  # doctest: +NORMALIZE_WHITESPACE,+ELLIPSIS
SMIRNOFFImproperTorsionCollection(type='ImproperTorsions',
    expression='k*(1+cos(periodicity*theta-phase))',
    key_map={},
    potentials={})

```

In the bond handler for example, each pair of bonded atoms maps to one of two
potential keys, one for the carbon-carbon bond, and the other for the
carbon-hydrogen bonds. It's clear from the SMIRKS codes that atoms 0 and 1 are
the carbon atoms, and atoms 2 through 7 are the hydrogens:

```pycon
>>> interchange.handlers['Bonds'].key_map  # doctest: +NORMALIZE_WHITESPACE,+ELLIPSIS
{TopologyKey(atom_indices=(0, 1), ...): PotentialKey(id='[#6X4:1]-[#6X4:2]', ...),
 TopologyKey(atom_indices=(0, 2), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...),
 TopologyKey(atom_indices=(0, 3), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...),
 TopologyKey(atom_indices=(0, 4), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...),
 TopologyKey(atom_indices=(1, 5), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...),
 TopologyKey(atom_indices=(1, 6), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...),
 TopologyKey(atom_indices=(1, 7), ...): PotentialKey(id='[#6X4:1]-[#1:2]', ...)}

```

:::{admonition} Question
Which atom indices represent hydrogens bonded to carbon atom 0, and which are
bonded to carbon atom 1?
:::

The bond handler also maps the two potential keys to the appropriate `Potential`.
Here we can read off the force constant and length:

```pycon
>>> interchange.handlers['Bonds'].potentials  # doctest: +NORMALIZE_WHITESPACE,+ELLIPSIS
{PotentialKey(id='[#6X4:1]-[#6X4:2]', ...):
     Potential(parameters={'k': <Quantity(529.242972, 'kilocalorie / angstrom ** 2 / mole')>,
                           'length': <Quantity(1.52190126, 'angstrom')>}, ...),
 PotentialKey(id='[#6X4:1]-[#1:2]', ...):
     Potential(parameters={'k': <Quantity(740.093414, 'kilocalorie / angstrom ** 2 / mole')>,
                           'length': <Quantity(1.09389949, 'angstrom')>}, ...)}

```

We can even modify a value here, export the new interchange, and see that all of
the bonds have been updated:

```pycon
>>> from openff.interchange.models import TopologyKey
>>> from openff.units import unit
>>> # Get the potential from the first C-H bond
>>> top_key = TopologyKey(atom_indices=(0, 2))
>>> pot_key = interchange.handlers['Bonds'].key_map[top_key]
>>> potential = interchange.handlers['Bonds'].potentials[pot_key]
>>> # Modify the potential
>>> potential.parameters['length'] = 3.1415926 * unit.nanometer
>>> # Write out the modified interchange to a GROMACS .top file
>>> interchange.to_top("out.top")
>>> with open("out.top") as f:
...     print(f.read())  # doctest: +NORMALIZE_WHITESPACE,+ELLIPSIS
; Generated by Interchange
    ...
[ bonds ]
;     ai      aj  func  r              k
      1       2   1     0.152190126495 221435.2592902858
      1       3   1     3.1415926      309655.084322414
      1       4   1     3.1415926      309655.084322414
      1       5   1     3.1415926      309655.084322414
      2       6   1     3.1415926      309655.084322414
      2       7   1     3.1415926      309655.084322414
      2       8   1     3.1415926      309655.084322414
    ...

```

[`Collection`]: openff.interchange.components.potentials.Collection
[`TopologyKey`]: openff.interchange.models.TopologyKey
[`PotentialKey`]: openff.interchange.models.PotentialKey
[`Potential`]: openff.interchange.components.potentials.Potential
[`Interchange.handlers`]: openff.interchange.Interchange.topology
[`openff.interchange.smirnoff`]: openff.interchange.smirnoff
[`openff.interchange.foyer`]: openff.interchange.foyer
