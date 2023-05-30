"""
A wrapper around PACKMOL. Taken from OpenFF Evaluator v0.4.3.
"""
import os
import shutil
import subprocess
import tempfile
from distutils.spawn import find_executable
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from openff.toolkit import Molecule, RDKitToolkitWrapper, Topology
from openff.units import Quantity, unit
from openff.utilities.utilities import requires_package, temporary_cd

from openff.interchange.exceptions import PACKMOLRuntimeError, PACKMOLValueError

UNIT_CUBE = np.asarray(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
)
"""A regular square prism with image distance 1.0 and volume 1.0.

A cubic box is very simple and easy to visualize, but wastes substantial space
and computational time. A rhombic dodecahedron that provides the same separation
between solutes has only about 71% the volume, and therefore requires simulation
of much less solvent.
"""

RHOMBIC_DODECAHEDRON = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, np.sqrt(2.0) / 2.0],
    ],
)
"""
Rhombic dodecahedron with square XY plane cross section, image distance 1.0, and volume ~0.71.

The rhombic dodecahedron is the most space-efficient triclinic box for a
spherical solute, or equivalently for a solute whose rotations sweep out a
sphere. Its volume is $\\frac{1}{2}\\sqrt{2}$. A square intersection with the
XY plane allows the first two box vectors to be parallel to the x- and y-axes
respectively, which is simple to picture and appropriate for soluble systems.
"""

RHOMBIC_DODECAHEDRON_XYHEX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0, 0.0],
        [0.5, np.sqrt(3.0) / 6.0, np.sqrt(6.0) / 3.0],
    ],
)
"""
A rhombic dodecahedron with hexagonal XY cross section, image distance 1.0, and volume ~0.71.

The rhombic dodecahedron is the most space-efficient triclinic box for a
spherical solute, or equivalently for a solute whose rotations sweep out a
sphere. A hexagonal intersection with the XY plane is convenient for membrane
simulations,
"""


def _find_packmol() -> Optional[str]:
    """
    Attempt to find the path to the `packmol` binary.

    Returns
    -------
    str, optional
        The path to the packmol binary if it could be found, otherwise
        `None`.

    """
    return (
        find_executable("packmol") or shutil.which("packmol") or None
        if "PACKMOL" not in os.environ
        else os.environ["PACKMOL"]
    )


def _validate_inputs(
    molecules: list[Molecule],
    number_of_copies: list[int],
    solute: Optional[Topology],
    box_shape: NDArray,
    box_vectors: Optional[Quantity],
    mass_density: Optional[Quantity],
):
    """
    Validate the inputs which were passed to the main pack method.

    Parameters
    ----------
    molecules : list of openff.toolkit.topology.Molecule
        The molecules in the system.
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of `molecules`.
    solute: Topology, optional
        The OpenFF Topology to be solvated.
    box_vectors : openff.units.Quantity,
        The box vectors to fill in units compatible with angstroms. If `None`,
        `mass_density` must be provided.
    mass_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
         If `None`, `box_size` must be provided.
    box_shape: NDArray
        The shape of the simulation box, used in conjunction with the
        `mass_density` parameter. Should have shape (3, 3) with all positive
        elements.

    """
    if (
        box_vectors is None
        and mass_density is None  # noqa: W503
        and (solute is None or solute.box_vectors is None)  # noqa: W503
    ):
        raise PACKMOLValueError(
            "One of `box_vectors`, `mass_density`, or"
            + " `solute.box_vectors` must be specified.",  # noqa: W503
        )
    if box_vectors is not None and mass_density is not None:
        raise PACKMOLValueError(
            "`box_vectors` and `mass_density` cannot be specified together;"
            + " choose one or the other.",  # noqa: W503
        )

    if box_vectors is not None and box_vectors.shape != (3, 3):
        raise PACKMOLValueError(
            "`box_vectors` must be a openff.units.unit.Quantity Array with shape (3, 3)",
        )

    if box_shape.shape != (3, 3):
        raise PACKMOLValueError(
            "`box_shape` must be an array with shape (3, 3) or (3,)",
        )
    if not np.all(np.linalg.norm(box_shape, axis=-1) > 0.0):
        raise PACKMOLValueError("All vectors in `box_shape` must have a positive norm.")

    if len(molecules) != len(number_of_copies):
        raise PACKMOLValueError(
            "The length of `molecules` and `number_of_copies` must be identical.",
        )

    if solute is not None:
        if not isinstance(solute, Topology):
            raise PACKMOLValueError(
                "`solute` must be a openff.toolkit.topology.Topology",
            )

        positions = solute.get_positions()

        try:
            assert positions is not None
            assert positions.shape[0] == solute.n_atoms
        except AssertionError:
            raise PACKMOLValueError(
                "`solute` missing some atomic positions.",
            )


def _unit_vec(vec: Quantity) -> Quantity:
    """Get a unit vector in the direction of ``vec``."""
    return vec / np.linalg.norm(vec)


def _compute_brick_from_box_vectors(box_vectors: Quantity) -> Quantity:
    """
    Compute the rectangular brick for the given triclinic box vectors.

    This function implements Eqn 6 in:

    https://doi.org/10.1002/(SICI)1096-987X(19971130)18:15%3C1930::AID-JCC8%3E3.0.CO;2-P

    Parameters
    ----------
    box_vectors: NDArray
        Array with shape (3, 3) representing the box vectors of a triclinic cell

    """
    working_unit = box_vectors.u
    k, l, m = box_vectors.m

    # Compute some re-used intermediates
    k_hat = _unit_vec(k)
    k_cross_hat_l = _unit_vec(np.cross(k, l))

    # Compute the UVW representation
    u = k
    v = l - np.dot(l, k_hat) * k_hat
    w = np.dot(m, k_cross_hat_l) * k_cross_hat_l

    # Make sure the UVW representation is rectangular - if it isn't, that's a bug
    assert np.all(u + v + w == (u[0], v[1], w[2]))

    return np.asarray([u[0], v[1], w[2]]) * working_unit


def _range_neg_pos(stop):
    """Yield 0, 1, -1, 2, -2... ``stop - 1``, ``-(stop - 1)``."""
    yield 0
    for i in range(1, stop):
        yield i
        yield -i


def _iter_lattice_vecs(box, max_order):
    """Yield linear combinations of box vectors until max_order is reached."""
    a, b, c = box
    for i in _range_neg_pos(max_order + 1):
        for j in _range_neg_pos(max_order + 1):
            for k in _range_neg_pos(max_order + 1):
                yield i * a + j * b + k * c


def _wrap_into(
    points: Quantity,
    box: Quantity,
    condition: Callable,
    max_order: int = 3,
):
    """
    Convert a triclinic box to a representation where all atoms satisfy a condition.

    Parameters
    ----------
    points
        The points to transform
    box
        The triclinic box vectors
    condition
        The condition that must be satisfied. This should be a function taking
        an array of positions with shape (n, 3) and returning an array of
        booleans with shape (n,) whose elements are ``True`` at positions that
        satisfy the condition.
    max_order
        The maximum number of box vectors that points may be from the rectangular
        box.

    """
    assert points.shape[-1] == 3
    assert box.shape == (3, 3)
    points = points.copy()

    # Iterate over linear combinations of lattice vectors
    for lattice_vec in _iter_lattice_vecs(box, max_order):
        # If all the points satisfy the condition, we're done
        if np.all(condition(points)):
            break

        # Otherwise, choose the points that would satisfy the condition if we
        # translated them by the current linear combination of lattice vectors,
        # and do so
        translated_points = points - lattice_vec
        correct_translated_points = condition(translated_points)
        points -= correct_translated_points[:, None] * lattice_vec[None, :]
    else:
        raise PACKMOLValueError(
            f"Couldn't move all particles to satisfy condition in {max_order} steps",
        )

    return points


def _wrap_into_brick(points: Quantity, box: Quantity, max_order: int = 3):
    """
    Convert a triclinic box to its rectangular brick representation.

    Parameters
    ----------
    points
        The points to transform
    box
        The triclinic box vectors
    max_order
        The maximum number of box vectors that points may be from the rectangular
        box.

    """
    brick = _compute_brick_from_box_vectors(box)
    return _wrap_into(
        points,
        box,
        lambda points: np.all((np.zeros(3) <= points) & (points < brick), axis=-1),
        max_order,
    )


def _box_from_density(
    molecules: list[Molecule],
    n_copies: list[int],
    mass_density: Quantity,
    box_shape: NDArray,
) -> Quantity:
    """
    Approximate box size.

    Generate an approximate box size based on the number and molecular
    weight of the molecules present, and a target density for the final
    solvated mixture.

    Parameters
    ----------
    molecules : list of openff.toolkit.topology.Molecule
        The molecules in the system.
    n_copies : list of int
        The number of copies of each molecule.
    mass_density : openff.units.Quantity
        The target mass density for final system. It should have units
        compatible with g / mL.
    box_shape: NDArray
        The shape of the simulation box, used in conjunction with the
        `mass_density` parameter. Should have shape (3, 3) with all positive
        elements.

    Returns
    -------
    box_vectors: openff.units.Quantity
        The unit cell box vectors. Array with shape (3, 3)

    """
    working_unit = unit.angstrom
    # Get the desired volume in cubic working units
    total_mass = sum(
        sum([atom.mass for atom in molecule.atoms]) * n
        for molecule, n in zip(molecules, n_copies)
    )
    volume = (total_mass / mass_density).m_as(working_unit**3)

    # Scale the box shape so that it has unit volume
    box_shape_volume = np.linalg.det(box_shape)

    # Scale the box up to the desired volume
    box_vectors = volume * box_shape / box_shape_volume

    return box_vectors * working_unit


def _create_solute_pdb(
    topology: Optional[Topology],
    box_vectors: Quantity,
) -> Optional[str]:
    """Write out the solute topology to PDB so that packmol can read it."""
    if topology is None:
        return None

    # Copy the topology so we can change it
    topology = Topology(topology)
    # Wrap the positions into the brick representation so that packmol
    # sees all of them
    topology.set_positions(
        _wrap_into_brick(
            topology.get_positions(),
            box_vectors,
        ),
    )
    # Write to pdb
    solute_pdb_filename = "solvate.pdb"
    topology.to_file(
        solute_pdb_filename,
        file_format="PDB",
    )
    return solute_pdb_filename


def _create_molecule_pdbs(molecules: list[Molecule]) -> list[str]:
    """Write out PDBs of the molecules so that packmol can read them."""
    pdb_file_names = []
    for index, molecule in enumerate(molecules):
        # Make a copy of the molecule so we don't change the input
        molecule = Molecule(molecule)

        # Generate conformers if they're missing
        if molecule.n_conformers <= 0:
            molecule.generate_conformers(n_conformers=1)

        # RDKitToolkitWrapper is less buggy than OpenEye for writing PDBs
        # See https://github.com/openforcefield/openff-toolkit/issues/1307
        # and linked issues. As long as the PDB files have the atoms in the
        # right order, we'll be able to read packmol's output, since we
        # just load the PDB coordinates into a topology created from its
        # component molecules.
        pdb_file_name = f"{index}.pdb"
        pdb_file_names.append(pdb_file_name)

        molecule.to_file(
            pdb_file_name,
            file_format="PDB",
            toolkit_registry=RDKitToolkitWrapper(),
        )
    return pdb_file_names


def _build_input_file(
    molecule_file_names: list[str],
    molecule_counts: list[int],
    structure_to_solvate: Optional[str],
    box_size: Quantity,
    tolerance: Quantity,
) -> tuple[str, str]:
    """
    Construct the packmol input file.

    Parameters
    ----------
    molecule_file_names: list of str
        The paths to the molecule pdb files.
    molecule_counts: list of int
        The number of each molecule to add.
    structure_to_solvate: str, optional
        The path to the structure to solvate.
    box_size: openff.units.Quantity
        The lengths of each side of the box we want to fill. This is the box
        size of the rectangular brick representation of the simulation box; the
        packmol box will be shrunk by the tolerance.
    tolerance: openff.units.Quantity
        The packmol convergence tolerance.

    Returns
    -------
    str
        The path to the input file.
    str
        The path to the output file.

    """
    box_size = (box_size - tolerance).m_as(unit.angstrom)
    tolerance = tolerance.m_as(unit.angstrom)

    # Add the global header options.
    output_file_path = "packmol_output.pdb"
    input_lines = [
        f"tolerance {tolerance:f}",
        "filetype pdb",
        f"output {output_file_path}",
        "",
    ]

    # Add the section of the molecule to solvate if provided.
    if structure_to_solvate is not None:
        input_lines.extend(
            [
                f"structure {structure_to_solvate}",
                "  number 1",
                "  fixed 0. 0. 0. 0. 0. 0.",
                "end structure",
                "",
            ],
        )

    # Add a section for each type of molecule to add.
    for file_name, count in zip(molecule_file_names, molecule_counts):
        input_lines.extend(
            [
                f"structure {file_name}",
                f"  number {count}",
                f"  inside box 0. 0. 0. {box_size[0]} {box_size[1]} {box_size[2]}",
                "end structure",
                "",
            ],
        )

    packmol_input = "\n".join(input_lines)

    # Write packmol input
    packmol_file_name = "packmol_input.txt"

    with open(packmol_file_name, "w") as file_handle:
        file_handle.write(packmol_input)

    return packmol_file_name, output_file_path


def _center_topology_at(
    center_solute: Union[bool, Literal["BOX_VECS", "ORIGIN", "BRICK"]],
    topology: Topology,
    box_vectors: Quantity,
    brick_size: Quantity,
) -> Topology:
    """Return a copy of the topology centered as requested."""
    if isinstance(center_solute, str):
        center_solute = center_solute.upper()
    topology = Topology(topology)

    if center_solute is False:
        return topology
    elif center_solute in [True, "BOX_VECS"]:
        new_center = box_vectors.sum(axis=0) / 2.0
    elif center_solute == "ORIGIN":
        new_center = np.zeros(3)
    elif center_solute == "BRICK":
        new_center = brick_size / 2.0
    else:
        PACKMOLValueError(
            f"center_solute must be a bool, 'BOX_VECS', 'ORIGIN', or 'BRICK', not {center_solute!r}",
        )

    positions = topology.get_positions()
    center_of_geometry = positions.sum(axis=0) / len(positions)
    topology.set_positions(new_center - center_of_geometry + positions)
    return topology


@requires_package("rdkit")
def pack_box(
    molecules: list[Molecule],
    number_of_copies: list[int],
    solute: Optional[Topology] = None,
    tolerance: Quantity = 2.0 * unit.angstrom,
    box_vectors: Optional[Quantity] = None,
    mass_density: Optional[Quantity] = None,
    box_shape: ArrayLike = RHOMBIC_DODECAHEDRON,
    center_solute: Union[bool, Literal["BOX_VECS", "ORIGIN", "BRICK"]] = False,
    working_directory: Optional[str] = None,
    retain_working_files: bool = False,
) -> Topology:
    """
    Run packmol to generate a box containing a mixture of molecules.

    Parameters
    ----------
    molecules : list of openff.toolkit.topology.Molecule
        The molecules in the system.
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of ``molecules``.
    solute: Topology, optional
        An OpenFF :py:class:`Topology <openff.toolkit.topology.Topology>` to
        include in the box. If ``box_vectors`` and ``mass_density`` are not
        specified, box vectors can be taken from ``solute.box_vectors``.
    tolerance : openff.units.Quantity
        The minimum spacing between molecules during packing in units of
        distance. The default is large so that added waters do not disrupt the
        structure of proteins; when constructing a mixture of small molecules,
        values as small as 0.5 Å will converge faster and can still produce
        stable simulations after energy minimisation.
    box_vectors : openff.units.Quantity, optional
        The box vectors to fill in units of distance. If ``None``,
        ``mass_density`` must be provided. Array with shape (3,3).
    mass_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
        If ``None``, ``box_size`` must be provided.
    box_shape: Arraylike, optional
        The shape of the simulation box, used in conjunction with
        the ``mass_density`` parameter. Should be a dimensionless array with
        shape (3,3) for a triclinic box or (3,) for a rectangular box.
    center_solute
        How to center ``solute`` in the simulation box. If ``True``
        or ``"box_vecs"``, the solute's center of geometry will be placed at
        the center of the box's parallelopiped representation. If ``"origin"``,
        the solute will centered at the origin. If ``"brick"``, the solute will
        be centered in the box's rectangular brick representation. If
        ``False`` (the default), the solute will not be moved.
    working_directory: str, optional
        The directory in which to generate the temporary working files. If
        ``None``, a temporary one will be created.
    retain_working_files: bool
        If ``True`` all of the working files, such as individual molecule
        coordinate files, will be retained.

    Returns
    -------
    Topology
        An OpenFF ``Topology`` with the solvated system.

    Raises
    ------
    PACKMOLRuntimeError
        When packmol fails to execute / converge.

    """
    import rdkit

    # Make sure packmol can be found.
    packmol_path = _find_packmol()

    if packmol_path is None:
        raise OSError("Packmol not found, cannot run pack_box()")

    box_shape = np.asarray(box_shape)
    if box_shape.shape == (3,):
        box_shape = box_shape * np.identity(3)

    # Validate the inputs.
    _validate_inputs(
        molecules,
        number_of_copies,
        solute,
        box_shape,
        box_vectors,
        mass_density,
    )

    # Estimate the box_size from mass density if one is not provided.
    if mass_density is not None:
        box_vectors = _box_from_density(
            molecules,
            number_of_copies,
            mass_density,
            box_shape,
        )
    # If neither box size nor density are given, take box vectors from solute
    # topology
    if box_vectors is None:
        box_vectors = solute.box_vectors

    # Compute the dimensions of the equivalent brick - this is what packmol will
    # fill
    brick_size = _compute_brick_from_box_vectors(box_vectors)

    # Center the solute
    if center_solute and solute is not None:
        solute = _center_topology_at(
            center_solute,
            solute,
            box_vectors,
            brick_size,
        )

    # Set up the directory to create the working files in.
    temporary_directory = False
    if working_directory is None:
        working_directory = tempfile.mkdtemp()
        temporary_directory = True

    if len(working_directory) > 0:
        os.makedirs(working_directory, exist_ok=True)

    with temporary_cd(working_directory):
        solute_pdb_filename = _create_solute_pdb(
            solute,
            box_vectors,
        )

        # Create PDB files for all of the molecules.
        pdb_file_names = _create_molecule_pdbs(molecules)

        # Generate the input file.
        input_file_path, output_file_path = _build_input_file(
            pdb_file_names,
            number_of_copies,
            solute_pdb_filename,
            brick_size,
            tolerance,
        )

        with open(input_file_path) as file_handle:
            result = subprocess.check_output(
                packmol_path,
                stdin=file_handle,
                stderr=subprocess.STDOUT,
            ).decode("utf-8")

            packmol_succeeded = result.find("Success!") > 0

        if not packmol_succeeded:
            raise PACKMOLRuntimeError(result)

        # Load the coordinates from the PDB file with RDKit (because its already
        # a dependency)
        rdmol = rdkit.Chem.rdmolfiles.MolFromPDBFile(
            output_file_path,
            sanitize=False,
            removeHs=False,
            proximityBonding=False,
        )
        positions = rdmol.GetConformers()[0].GetPositions()

    # TODO: This currently does not run if we encountered an error in the
    # context manager
    if temporary_directory and not retain_working_files:
        shutil.rmtree(working_directory)

    # Construct the output topology
    added_molecules = []
    for mol, n in zip(molecules, number_of_copies):
        added_molecules.extend([mol] * n)
    topology = Topology.from_molecules(added_molecules)

    # Set the positions, skipping the positions from solute
    n_solute_atoms = len(positions) - topology.n_atoms
    topology.set_positions(positions[n_solute_atoms:] * unit.angstrom)

    # Add solute back in with the original, unwrapped positions
    if solute is not None:
        topology = solute + topology

    # Set the box vectors
    topology.box_vectors = box_vectors

    return topology


def _max_dist_between_points(points: Quantity) -> Quantity:
    """
    Compute the greatest distance between two points in the array.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist

    points, units = points.m, points.u

    points_array = np.asarray(points)
    if points_array.shape[1] != 3 or points_array.ndim != 2:
        raise PACKMOLValueError("Points should be an n*3 array")

    if points_array.shape[0] >= 4:
        # Maximum distance is guaranteed to be on the convex hull, which can be
        # computed in O(n log n)
        # See also https://stackoverflow.com/a/60955825
        hull = ConvexHull(points_array)
        hullpoints = points_array[hull.vertices, :]
    else:
        hullpoints = points_array

    # Now compute all the distances and get the greatest distance in O(h^2)
    max_dist = pdist(hullpoints, metric="euclidean").max()
    return max_dist * units


def solvate_topology(
    topology: Topology,
    nacl_conc: Quantity = 0.1 * unit.mole / unit.liter,
    padding: Quantity = 1.2 * unit.nanometer,
    box_shape: NDArray = RHOMBIC_DODECAHEDRON,
    target_density: Quantity = 1.0 * unit.gram / unit.milliliter,
    tolerance: Quantity = 2.0 * unit.angstrom,
) -> Topology:
    """
    Add water and ions to neutralise and solvate a topology.

    Parameters
    ----------
    topology
        The OpenFF Topology to solvate.
    nacl_conc
        The bulk concentration of NaCl in the solvent, in units compatible with
        molarity. This is used to calculate a mass fraction for the bulk
        solvent and does not represent the actual concentration in the final
        box.
    padding : Scalar with dimensions of length
        The desired distance between the solute and the edge of the box. Ignored
        if the topology already has box vectors. The usual recommendation is
        that this equals or exceeds the VdW cut-off distance, so that the
        solute is isolated by its periodic images by twice the cut-off.
    box_shape : Array with shape (3, 3)
        An array defining the box vectors of a box with the desired shape and
        unit periodic image distance. This shape will be scaled to satisfy the
        padding given above. Some typical shapes are provided in this module.
    target_density : Scalar with dimensions of mass density
        The target mass density for the packed box.
    tolerance: Scalar with dimensions of distance
        The minimum spacing between molecules during packing in units of
        distance. The default is large so that added waters do not disrupt the
        structure of proteins; when constructing a mixture of small molecules,
        values as small as 0.5 Å will converge faster and can still produce
        stable simulations after energy minimisation.

    """
    if box_shape.shape != (3, 3):
        raise PACKMOLValueError(
            "box_shape should be a 3×3 array defining a box with unit periodic"
            + " image distance",  # noqa: W503
        )

    # Compute box vectors from the solute length and requested padding
    solute_length = _max_dist_between_points(topology.get_positions())
    image_distance = solute_length + padding * 2
    box_vectors = box_shape * image_distance

    # Compute target masses of solvent
    box_volume = np.linalg.det(box_vectors.m) * box_vectors.u**3
    target_mass = box_volume * target_density
    solute_mass = sum(
        sum([atom.mass for atom in molecule.atoms]) for molecule in topology.molecules
    )
    solvent_mass = target_mass - solute_mass
    if solvent_mass < 0:
        raise PACKMOLValueError(
            "Solute mass is greater than target mass; increase density or make the box bigger",
        )

    # Get reference data and prepare solvent molecules
    water = Molecule.from_smiles("O")
    na = Molecule.from_smiles("[Na+]")
    cl = Molecule.from_smiles("[Cl-]")
    nacl_mass = sum([atom.mass for atom in na.atoms]) + sum(
        [atom.mass for atom in cl.atoms],
    )
    water_mass = sum([atom.mass for atom in water.atoms])
    molarity_pure_water = 55.5 * unit.mole / unit.liter

    # Compute the number of salt "molecules" to add from the mass and concentration
    nacl_mass_fraction = (nacl_conc * nacl_mass) / (molarity_pure_water * water_mass)
    nacl_mass_to_add = solvent_mass * nacl_mass_fraction
    nacl_to_add = (nacl_mass_to_add / nacl_mass).m_as(unit.dimensionless).round()

    # Compute the number of water molecules to add to make up the remaining mass
    water_mass_to_add = solvent_mass - nacl_mass
    water_to_add = (water_mass_to_add / water_mass).m_as(unit.dimensionless).round()

    # Neutralise the system by adding and removing salt
    solute_charge = sum([molecule.total_charge for molecule in topology.molecules])
    na_to_add = np.ceil(nacl_to_add - solute_charge.m / 2.0)
    cl_to_add = np.floor(nacl_to_add + solute_charge.m / 2.0)

    # Pack the box
    return pack_box(
        [water, na, cl],
        [int(water_to_add), int(na_to_add), int(cl_to_add)],
        solute=topology,
        tolerance=2.0 * unit.angstrom,
        box_vectors=box_vectors,
    )
