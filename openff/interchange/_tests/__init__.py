"""Assorted utilities used in tests."""

import pathlib
import sys
from typing import Optional

import pytest
from openff.toolkit import Molecule
from openff.toolkit.utils import (
    AmberToolsToolkitWrapper,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
)
from openff.utilities.utilities import has_executable, has_package

from openff.interchange.drivers.gromacs import _find_gromacs_executable

if sys.version_info >= (3, 10):
    from importlib import resources
else:
    import importlib_resources as resources


if has_package("openmm"):
    import openmm
    import openmm.unit

    kj_nm2_mol = openmm.unit.kilojoule_per_mole / openmm.unit.nanometer**2
    kj_rad2_mol = openmm.unit.kilojoule_per_mole / openmm.unit.radian**2

requires_ambertools = pytest.mark.skipif(
    not AmberToolsToolkitWrapper.is_available(),
    reason="Test requires AmberTools",
)
requires_rdkit = pytest.mark.skipif(
    not RDKitToolkitWrapper.is_available(),
    reason="Test requires RDKit",
)
requires_openeye = pytest.mark.skipif(
    not OpenEyeToolkitWrapper.is_available(),
    reason="Test requires OE toolkit",
)


def get_test_file_path(test_file: str) -> pathlib.Path:
    """Given a filename in the collection of data files, return its full path."""
    test_dir_path = get_test_files_dir_path()
    test_file_path = test_dir_path / test_file

    if test_file_path.is_file():
        return test_file_path
    else:
        raise FileNotFoundError(
            f"could not file file {test_file} in path {test_file_path}",
        )


def get_test_files_dir_path(dirname: Optional[str] = None) -> pathlib.Path:
    """Given a directory with a collection of test data files, return its full path."""
    dir_path = resources.files("openff.interchange._tests.data")

    if dirname:
        test_dir: pathlib.PosixPath = dir_path / dirname  # type: ignore[assignment]
    else:
        test_dir = dir_path  # type: ignore[assignment]

    if test_dir.is_dir():
        return test_dir
    else:
        raise NotADirectoryError(
            f"Provided directory {dirname} doesn't exist in {dir_path}",
        )


class MoleculeWithConformer(Molecule):
    """Thin wrapper around `Molecule` to produce an instance with a conformer in one call."""

    @classmethod
    def from_smiles(self, smiles, name="", **kwargs):
        """Create from smiles and generate a single conformer."""
        molecule = super().from_smiles(smiles, **kwargs)
        molecule.generate_conformers(n_conformers=1)
        molecule.name = name

        return molecule

    @classmethod
    def from_mapped_smiles(self, smiles, name="", **kwargs):
        """Create from smiles and generate a single conformer."""
        molecule = super().from_mapped_smiles(smiles, **kwargs)
        molecule.generate_conformers(n_conformers=1)
        molecule.name = name

        return molecule


HAS_GROMACS = _find_gromacs_executable() is not None
HAS_LAMMPS = has_package("lammps")
HAS_SANDER = has_executable("sander")

needs_gmx = pytest.mark.skipif(not HAS_GROMACS, reason="Needs GROMACS")
needs_not_gmx = pytest.mark.skipif(
    HAS_GROMACS,
    reason="Needs GROMACS to NOT be installed",
)
needs_lmp = pytest.mark.skipif(not HAS_LAMMPS, reason="Needs LAMMPS")
needs_not_lmp = pytest.mark.skipif(
    HAS_LAMMPS,
    reason="Needs LAMMPS to NOT be installed",
)
needs_sander = pytest.mark.skipif(not HAS_SANDER, reason="Needs sander")
needs_not_sander = pytest.mark.skipif(
    HAS_SANDER,
    reason="sander needs to NOT be installed",
)
