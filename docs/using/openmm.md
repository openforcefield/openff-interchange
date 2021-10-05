# Use with OpenMM

An `Interchange` object can be converted to an `openmm.System` object with
`Interchange.to_openmm()`.

```python
openmm_sys = interchange.to_openmm()
```

By default, this will separate non-bonded interactions into several different `openmm.Force`
objects. To combine everything into a single `openmm.NonbondedForce`, use the
`combine_nonbonded_forces=True` argument.
