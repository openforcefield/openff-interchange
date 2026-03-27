"""Assorted utilities used in tests."""

import pathlib
import random
from importlib import resources

import numpy
import pytest
from openff.toolkit import (
    AmberToolsToolkitWrapper,
    ForceField,
    Molecule,
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
    Topology,
)
from openff.utilities import get_data_file_path
from openff.utilities.utilities import has_executable, has_package

from openff.interchange import Interchange
from openff.interchange.drivers.gromacs import _find_gromacs_executable

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


_rng = numpy.random.default_rng(12345)


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


def get_test_files_dir_path(dirname: str | None = None) -> pathlib.Path:
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
    def from_smiles(self, smiles, name="", *args, **kwargs):
        """Create from smiles and generate a single conformer."""
        molecule = super().from_smiles(smiles, *args, **kwargs)
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


def get_protein(name: str) -> Molecule:
    """Get a protein from openff/toolkit/data/proteins based on PDB name."""
    return Topology.from_pdb(
        get_data_file_path(
            relative_path=f"proteins/{name}.pdb",
            package_name="openff.toolkit",
        ),
    ).molecule(0)


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


def shuffle_topology(
    interchange: Interchange,
    force_field: ForceField,
    seed: int = 123455,
) -> Interchange:
    """Shuffle the atom order of the topology and return a new Interchange object."""
    assert interchange.positions is not None, "Written to also manage positions"

    rng = random.Random(seed)

    shuffled_topology = Topology()

    # can't easily shuffle a generator ...
    for molecule in interchange.topology.molecules:
        topology_indices = [interchange.topology.atom_index(atom) for atom in molecule.atoms]

        molecule._conformers = [
            interchange.positions[topology_indices, :].to("angstrom"),
        ]

        # so just randomize order at insertion time
        shuffled_topology._molecules.insert(
            rng.randint(0, len(shuffled_topology._molecules)),
            molecule,
        )

    shuffled_topology._invalidate_cached_properties()

    shuffled_topology.box_vectors = interchange.topology.box_vectors

    return force_field.create_interchange(shuffled_topology)
