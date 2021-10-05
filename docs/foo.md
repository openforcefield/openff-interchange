# Content of an Interchange object

There are four components of an `Interchange` object:
1. Topology
1. Handlers
2. Positions
3. Box vectors

None are strictly required; an `Interchange` object can beconstructed containing none of the above components:

```python3
>>> from openff.interchange.components.interchange import Interchange
>>> empty_interchange = Interchange()
>>> empty_interchange.topology
>>> empty_interchange.handlers
{}
>>> empty_interchange.positions
>>> empty_interchange.box
```

This is not useful in this state, but each component can assigned, allowing for programmitic
construction. (However, it is recommended that the API be used where possible, i.e.
`Interchange.from_smirnoff` and `Interchange.from_foyer`).

## Topology

The `Interchange.topology` object is inteded to mirror the
[`Topology`](https://open-forcefield-toolkit.readthedocs.io/en/stable/api/generated/openff.toolkit.topology.Topology.html#openff.toolkit.topology.Topology)
class provided by the OpenFF Toolkit. At present, however, it serves only as a container for a class
of the same name provided by
[MDTraj](https://www.mdtraj.org/1.9.5/api/generated/mdtraj.Topology.html#mdtraj.Topology), stored as
an attribute `Interchange.topology.mdtop`.

A topology is optional, though any conversions requiring topological information will fail if any required data is missing.

## Handlers

This attribute is a dictionary of key-value pairs in which the keys are string identifiers and the values are `PotentialHandler` objects.
These handlers roughly mirror
[`ParameterHandler`](https://open-forcefield-toolkit.readthedocs.io/en/stable/users/developing.html#parameterhandler) 
the necessary handler(s) are required.

Handlers are optional, though any conversions requiring force parameters will fail if the necessary handler(s) are required.

## Positions

Particle positions are stored as a unit-tagged Nx3 matrix where N is the number of particles in the
topology. For systems with no virtual sites, N is the number of atoms; for systems with virtual
sites, N is the number of atoms plus the number of virtual sites.

Positions are by represented with nanometers internally and when requested via the getter. However,
they can be set with positions of other units, and can also be converted as desired.

If positions are passed to the setter without tagged units, nanometers will be assumed. The setter
will immediately tag the unitless vectors with units; the internal state always has units explicitly
associated with with the values.

Positions are optional, though any conversions requiring positions will fail if they are missing.

```python3
>>> from openff.interchange.components.interchange import Interchange
>>> from openff.units import unit
>>> from openff.toolkit.topology import Molecule
>>> molecule = Molecule.from_smiles("CCO")
>>> molecule.generate_conformers(n_conformers=1)
>>> molecule.conformers[0]
Quantity(value=array([[ 0.88165321, -0.04478118, -0.01474324],
       [-0.58171004, -0.37572459,  0.05098497],
       [-1.35004062,  0.75806983,  0.17615782],
       [ 1.26504668,  0.17421359,  1.01224746],
       [ 1.01649295,  0.87054063, -0.60898906],
       [ 1.47635802, -0.89454965, -0.39185017],
       [-0.78535559, -0.99682774,  0.96832828],
       [-0.83550563, -1.00354494, -0.81588946],
       [-1.08693898,  1.51260405, -0.3762466 ]]), unit=angstrom)
>>> model = Interchange()
>>> model.positions is None
True
>>> model.positions = molecule.conformers[0]
>>> model.positions
<Quantity([[ 0.08816532 -0.00447812 -0.00147432]
 [-0.058171   -0.03757246  0.0050985 ]
 [-0.13500406  0.07580698  0.01761578]
 [ 0.12650467  0.01742136  0.10122475]
 [ 0.10164929  0.08705406 -0.06089891]
 [ 0.1476358  -0.08945496 -0.03918502]
 [-0.07853556 -0.09968277  0.09683283]
 [-0.08355056 -0.10035449 -0.08158895]
 [-0.1086939   0.1512604  -0.03762466]], 'nanometer')>
>>> model.positions.m_as(unit.angstrom)
array([[ 0.88165321, -0.04478118, -0.01474324],
       [-0.58171004, -0.37572459,  0.05098497],
       [-1.35004062,  0.75806983,  0.17615782],
       [ 1.26504668,  0.17421359,  1.01224746],
       [ 1.01649295,  0.87054063, -0.60898906],
       [ 1.47635802, -0.89454965, -0.39185017],
       [-0.78535559, -0.99682774,  0.96832828],
       [-0.83550563, -1.00354494, -0.81588946],
       [-1.08693898,  1.51260405, -0.3762466 ]])
```

## Box vectors

Information about the periodic box of a system is stored as a unit-tagged 3x3 matrix, following
conventional periodic box vectors and the implementation in
[`OpenMM`](http://docs.openmm.org/latest/userguide/theory/05_other_features.html#periodic-boundary-conditions).

Box vectors are by represented with nanometers internally and when requested via the getter. However, they can be set with box vectors of other
units, and can also be converted as desired.

If box vectors are passed to the setter without tagged units, nanometers will be assumed. The setter
will immediately tag the unitless vectors with units; the internal state always has units explicitly
associated with with the values.

If a 1x3 matrix (array) is passed to the setter, it is assumed that these values correspond to the
legnths of a rectangular unit cell (`a_x`, `b_y`, `c_z`).

Box vectors are optional; if it is `None` it is implied that the `Interchange` object represents a
non-periodic system.

```python3
>>> from openff.interchange.components.interchange import Interchange
>>> from openff.units import unit
>>> import numpy as np
>>> model = Interchange()
>>> model.box is None
True
>>> model.box = np.eye(3) * 4 * unit.nanometer
>>> model.box
<Quantity([[4. 0. 0.]
 [0. 4. 0.]
 [0. 0. 4.]], 'nanometer')>
>>> model.box = [3, 4, 5]
<Quantity([[3. 0. 0.]
 [0. 4. 0.]
 [0. 0. 5.]], 'nanometer')>
>>> model.box = [28, 28, 28] * unit.angstrom
<Quantity([[2.8 0. 0.]
 [0. 2.8 0.]
 [0. 0. 2.8]], 'nanometer')>
>>> model.box.m_as(unit.angstrom)
array([[28.,  0.,  0.],
       [ 0., 28.,  0.],
       [ 0.,  0., 28.]])
```
