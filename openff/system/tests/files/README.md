Documenting how some of these files were generated

`10_ar.pdb`:
  - 
  ```python3
    >>> import mbuild as mb
    >>> ar = mb.Compound(name='Ar')
    >>> mol = mb.fill_box(ar, 10, box=[2, 2, 2])
    >>> mol.to_parmed().save('10_ar.pdb')
  ```
