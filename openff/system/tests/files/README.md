Documenting how some of these files were generated

`10_ar.pdb`:
  -
  ```python3
    >>> import mbuild as mb
    >>> ar = mb.Compound(name='Ar')
    >>> mol = mb.fill_box(ar, 10, box=[2, 2, 2])
    >>> mol.to_parmed().save('10_ar.pdb')
  ```

`parsley.offxml`:
  - This file is a based on a copy of `openff_unconstrained-1.0.0.offxml` (1/4/21)
  - The 1-4 scaling term in the Electrostatics handler is hard-coded to 0.83333 to replicate a bug in the OpenFF Toolkit
    - See https://github.com/openforcefield/opennff-toolkit/issues/408
    - Once this bug is fixed, a mainline force field loaded from the entry point should be used
