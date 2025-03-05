"""
A wrapper around PACKMOL. Adapted from OpenFF Evaluator v0.4.3.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import numpy
from numpy.typing import ArrayLike, NDArray
from openff.toolkit import Molecule, Quantity, RDKitToolkitWrapper, Topology
from openff.utilities.utilities import requires_package, temporary_cd

from openff.interchange.exceptions import PACKMOLRuntimeError, PACKMOLValueError

UNIT_CUBE = numpy.asarray(
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

RHOMBIC_DODECAHEDRON = numpy.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, numpy.sqrt(2.0) / 2.0],
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

RHOMBIC_DODECAHEDRON_XYHEX = numpy.array(
    [
        [1.0, 0.0, 0.0],
        [0.5, numpy.sqrt(3.0) / 2.0, 0.0],
        [0.5, numpy.sqrt(3.0) / 6.0, numpy.sqrt(6.0) / 3.0],
    ],
)
"""
A rhombic dodecahedron with hexagonal XY cross section, image distance 1.0, and volume ~0.71.

The rhombic dodecahedron is the most space-efficient triclinic box for a
spherical solute, or equivalently for a solute whose rotations sweep out a
sphere. A hexagonal intersection with the XY plane is convenient for membrane
simulations,
"""


def _find_packmol() -> str | None:
    """
    Attempt to find the path to the `packmol` binary.

    Returns
    -------
    str, optional
        The path to the packmol binary if it could be found, otherwise
        `None`.

    """
    return shutil.which("packmol")


def _check_add_positive_mass(mass_to_add):
    if mass_to_add.m < 0:
        raise PACKMOLValueError(
            "Solute mass is greater than target mass; increase density or make the box bigger",
        )


def _check_box_shape_shape(box_shape: NDArray):
    """Check the .shape of the box_shape argument."""
    if box_shape.shape != (3, 3):
        raise PACKMOLValueError(
            "box_shape should be an array with shape (3, 3) defining a box with unit periodic image distance",
        )


def _validate_inputs(
    molecules: list[Molecule],
    number_of_copies: list[int],
    solute: Topology | None,
    box_shape: NDArray,
    box_vectors: Quantity | None,
    target_density: Quantity | None,
):
    """
    Validate the inputs which were passed to the main pack method.

    Parameters
    ----------
    molecules : list of openff.toolkit.Molecule
        The molecules in the system.
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of `molecules`.
    solute: Topology, optional
        The OpenFF Topology to be solvated.
    box_vectors : openff.units.Quantity,
        The box vectors to fill in units compatible with angstroms. If `None`,
        `target_density` must be provided.
    target_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
         If `None`, `box_size` must be provided.
    box_shape: NDArray
        The shape of the simulation box, used in conjunction with the
        `target_density` parameter. Should have shape (3, 3) with all positive
        elements.

    """
    if box_vectors is None and target_density is None and (solute is None or solute.box_vectors is None):
        raise PACKMOLValueError(
            "One of `box_vectors`, `target_density`, or" + " `solute.box_vectors` must be specified.",
        )
    if box_vectors is not None and target_density is not None:
        raise PACKMOLValueError(
            "`box_vectors` and `target_density` cannot be specified together;" + " choose one or the other.",
        )

    if box_vectors is not None and box_vectors.shape != (3, 3):
        raise PACKMOLValueError(
            "`box_vectors` must be a openff.units.Quantity Array with shape (3, 3)",
        )

    if box_shape.shape != (3, 3):
        raise PACKMOLValueError(
            "`box_shape` must be an array with shape (3, 3) or (3,)",
        )
    if not numpy.all(numpy.linalg.norm(box_shape, axis=-1) > 0.0):
        raise PACKMOLValueError("All vectors in `box_shape` must have a positive norm.")

    if len(molecules) != len(number_of_copies):
        raise PACKMOLValueError(
            "The length of `molecules` and `number_of_copies` must be identical.",
        )

    if solute is not None:
        if not isinstance(solute, Topology):
            raise PACKMOLValueError(
                "`solute` must be a openff.toolkit.Topology",
            )

        positions = solute.get_positions()

        try:
            assert positions is not None
            assert positions.shape[0] == solute.n_atoms
        except AssertionError:
            raise PACKMOLValueError(
                "`solute` missing some atomic positions.",
            )


def _box_vectors_are_in_reduced_form(box_vectors: Quantity) -> bool:
    """
    Return ``True`` if the box is in OpenMM reduced form; ``False`` otherwise.

    These conditions are shared by OpenMM and GROMACS and greatly simplify
    working with triclinic boxes. Any periodic system can be represented in this
    form by rotating the system and lattice reduction.
    See http://docs.openmm.org/latest/userguide/theory/05_other_features.html#periodic-boundary-conditions
    """
    assert box_vectors.shape == (3, 3)
    a, b, c = box_vectors.m
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    return (
        [ay, az] == [0, 0]
        and bz == 0
        and ax > 0
        and by > 0
        and cz > 0
        and ax >= 2 * numpy.abs(bx)
        and ax >= 2 * numpy.abs(cx)
        and by >= 2 * numpy.abs(cy)
    )


def _unit_vec(vec: Quantity) -> Quantity:
    """Get a unit vector in the direction of ``vec``."""
    return vec / numpy.linalg.norm(vec)


def _compute_brick_from_box_vectors(box_vectors: Quantity) -> Quantity:
    """
    Compute the rectangular brick for the given triclinic box vectors.

    Parameters
    ----------
    box_vectors: NDArray
        Array with shape (3, 3) representing the box vectors of a triclinic cell

    """
    # This should have already been checked with a nice error message, but it is
    # an important invariant so we'll check it again here
    assert _box_vectors_are_in_reduced_form(box_vectors)
    return numpy.diagonal(box_vectors)


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
        if numpy.all(condition(points)):
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
        lambda points: numpy.all(
            (numpy.zeros(3) <= points) & (points < brick),
            axis=-1,
        ),
        max_order,
    )


def _box_from_density(
    molecules: list[Molecule],
    n_copies: list[int],
    target_density: Quantity,
    box_shape: NDArray,
) -> Quantity:
    """
    Approximate box size.

    Generate an approximate box size based on the number and molecular
    weight of the molecules present, and a target density for the final
    solvated mixture.

    Parameters
    ----------
    molecules : list of openff.toolkit.Molecule
        The molecules in the system.
    n_copies : list of int
        The number of copies of each molecule.
    target_density : openff.units.Quantity
        The target mass density for final system. It should have units
        compatible with g / mL.
    box_shape: NDArray
        The shape of the simulation box, used in conjunction with the
        `target_density` parameter. Should have shape (3, 3) with all positive
        elements.

    Returns
    -------
    box_vectors: openff.units.Quantity
        The unit cell box vectors. Array with shape (3, 3)

    """
    # Get the desired volume in cubic working units
    total_mass = sum(sum([atom.mass for atom in molecule.atoms]) * n for molecule, n in zip(molecules, n_copies))
    volume = total_mass / target_density

    return _scale_box(box_shape, volume)


def _scale_box(box: NDArray, volume: Quantity, box_scaleup_factor=1.1) -> Quantity:
    """
    Scale the parallelepiped spanned by ``box`` to the given volume.

    The volume of the parallelepiped spanned by the rows of a matrix is the
    determinant of that matrix, and scaling a row of a matrix by a constant c
    scales the determinant by that same constant; therefore scaling all three
    rows by c scales the volume by c**3.

    Parameters
    ----------
    box
        3x3 matrix whose rows are the box vectors.
    volume
        Desired scalar volume of the box in units of volume.

    box_scaleup_factor
        The factor, applied linearly, by which the estimated box size should be increased.

    Returns
    -------
    scaled_box
        3x3 matrix in angstroms.

    """
    final_volume = volume.m_as("angstrom ** 3")

    initial_volume = numpy.abs(numpy.linalg.det(box))
    volume_scale_factor = final_volume / initial_volume
    linear_scale_factor = volume_scale_factor ** (1 / 3)

    linear_scale_factor *= box_scaleup_factor

    return Quantity(linear_scale_factor * box, "angstrom")


def _create_solute_pdb(
    topology: Topology | None,
    box_vectors: Quantity,
) -> str | None:
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
    solute_pdb_filename = "_PACKING_SOLUTE.pdb"
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
        else:
            # Possible issues writing multi-conformer PDBs with RDKit
            # https://github.com/openforcefield/openff-interchange/issues/1107
            molecule._conformers = [molecule.conformers[0]]

        # RDKitToolkitWrapper is less buggy than OpenEye for writing PDBs
        # See https://github.com/openforcefield/openff-toolkit/issues/1307
        # and linked issues. As long as the PDB files have the atoms in the
        # right order, we'll be able to read packmol's output, since we
        # just load the PDB coordinates into a topology created from its
        # component molecules.
        pdb_file_name = f"_PACKING_MOLECULE{index}.pdb"
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
    structure_to_solvate: str | None,
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
    box_size = (box_size - tolerance).m_as("angstrom")
    tolerance = tolerance.m_as("angstrom")

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
    center_solute: bool | Literal["BOX_VECS", "ORIGIN", "BRICK"],
    topology: Topology,
    box_vectors: Quantity,
    brick_size: Quantity,
) -> Topology:
    """Return a copy of the topology centered as requested."""
    topology = Topology(topology)

    if isinstance(center_solute, str):
        center_solute = center_solute.upper()  # type: ignore[assignment]

    if center_solute is False:
        return topology
    elif center_solute in [True, "BOX_VECS"]:
        new_center = box_vectors.sum(axis=0) / 2.0
    elif center_solute == "ORIGIN":
        new_center = numpy.zeros(3)
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
    solute: Topology | None = None,
    tolerance: Quantity = Quantity(2.0, "angstrom"),
    box_vectors: Quantity | None = None,
    target_density: Quantity | None = None,
    box_shape: ArrayLike = RHOMBIC_DODECAHEDRON,
    center_solute: bool | Literal["BOX_VECS", "ORIGIN", "BRICK"] = False,
    working_directory: str | None = None,
    retain_working_files: bool = False,
) -> Topology:
    """
    Run packmol to generate a box containing a mixture of molecules.

    Parameters
    ----------
    molecules : list of openff.toolkit.Molecule
        The molecules in the system.
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of ``molecules``.
    solute: Topology, optional
        An OpenFF :py:class:`Topology <openff.toolkit.Topology>` to
        include in the box. If ``box_vectors`` and ``target_density`` are not
        specified, box vectors can be taken from ``solute.box_vectors``.
    tolerance : openff.units.Quantity
        The minimum spacing between molecules during packing in units of
        distance. The default is large so that added waters do not disrupt the
        structure of proteins; when constructing a mixture of small molecules,
        values as small as 0.5 Å will converge faster and can still produce
        stable simulations after energy minimisation.
    box_vectors : openff.units.Quantity, optional
        The box vectors to fill in units of distance. If ``None``,
        ``target_density`` must be provided. Array with shape (3,3). Box vectors
        must be provided in `OpenMM reduced form <http://docs.openmm.org/latest/
        userguide/theory/05_other_features.html#periodic-boundary-conditions>`_.
    target_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
        If ``None``, ``box_vectors`` must be provided.
    box_shape: Arraylike, optional
        The shape of the simulation box, used in conjunction with
        the ``target_density`` parameter. Should be a dimensionless array with
        shape (3,3) for a triclinic box or (3,) for a rectangular box. Shape
        vectors must be provided in `OpenMM reduced form
        <http://docs.openmm.org/latest/userguide/theory/
        05_other_features.html#periodic-boundary-conditions>`_.
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

    Notes
    -----
    Returned topologies may have smaller or larger box vectors than what would be defined by the
    target density if the box vectors are determined by `target_density`. When calling Packmol, each
    linear dimension of the box is scaled up by 10%.  However, Packmol by default adds a small
    buffer (defined by the `tolerance` argument which otherwise defines the minimum distance,
    default 2 Angstrom) at the end of the packed box, which causes small voids when tiling copies of
    each periodic image. This void is removed in hopes of faster equilibration times but the box
    density is slightly increased as a result. These changes may cancel each other out or result in
    larger or smaller densities than the target density, depending on argument values.

    """
    # Make sure packmol can be found.
    packmol_path = _find_packmol()

    if packmol_path is None:
        raise OSError("Packmol not found, cannot run pack_box()")

    box_shape = numpy.asarray(box_shape)
    if box_shape.shape == (3,):
        box_shape = box_shape * numpy.identity(3)

    # Validate the inputs.
    _validate_inputs(
        molecules,
        number_of_copies,
        solute,
        box_shape,
        box_vectors,
        target_density,
    )

    # Estimate the box_vectors from mass density if one is not provided.
    if target_density is not None:
        box_vectors = _box_from_density(
            molecules,
            number_of_copies,
            target_density,
            box_shape,
        )
    # If neither box vectors nor density are given, take box vectors from solute
    # topology
    if box_vectors is None:
        box_vectors = solute.box_vectors  # type: ignore[union-attr]

    if not _box_vectors_are_in_reduced_form(box_vectors):
        raise PACKMOLValueError(
            "pack_box requires box vectors to be in OpenMM reduced form.\n"
            + "See http://docs.openmm.org/latest/userguide/theory/"
            + "05_other_features.html#periodic-boundary-conditions",
        )

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
            try:
                result = subprocess.check_output(
                    packmol_path,
                    stdin=file_handle,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as error:
                # Custom error codes seem to start at 170
                # https://github.com/m3g/packmol/blob/v20.15.1/src/exit_codes.f90#L13-L16
                open("packmol_error.log", "w").write(error.stdout.decode("utf-8"))

                raise PACKMOLRuntimeError(
                    f"PACKMOL failed with error code {error.returncode}. Wrote file packmol_error.log in working "
                    "directory, which might be a temporary directory. Set the argument `working_directory` to "
                    "point this to a persistent path.",
                ) from error

            packmol_succeeded = result.decode("utf-8").find("Success!") > 0

        if not packmol_succeeded:
            raise PACKMOLRuntimeError(
                "PACKMOL did not raise an error code, but 'Success!' not found in output. "
                "Please raise an issue showing how you arrived at this error.",
            )

        positions = _load_positions(output_file_path)

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
    topology.set_positions(Quantity(positions[n_solute_atoms:], "angstrom"))

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
    from scipy.spatial import ConvexHull, distance

    points, units = points.m, points.u

    points_array = numpy.asarray(points)
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
    max_dist = distance.pdist(hullpoints, metric="euclidean").max()
    return max_dist * units


def _load_positions(output_file_path) -> NDArray:
    try:
        return numpy.asarray(
            [
                [line[31:39], line[39:46], line[47:54]]
                for line in open(output_file_path).readlines()
                if line.startswith("HETATM") or line.startswith("ATOM")
            ],
            dtype=numpy.float32,
        )
    except Exception as error:
        raise PACKMOLRuntimeError(
            "PACKMOL output could not be parsed by a native coordinate parser, "
            "please raise an issue with code reproducing this error.",
        ) from error


def solvate_topology(
    topology: Topology,
    nacl_conc: Quantity = Quantity(0.1, "mole / liter"),
    padding: Quantity = Quantity(1.2, "nanometer"),
    box_shape: NDArray = RHOMBIC_DODECAHEDRON,
    target_density: Quantity = Quantity(0.9, "gram / milliliter"),
    tolerance: Quantity = Quantity(2.0, "angstrom"),
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

    Returns
    -------
    Topology
        An OpenFF ``Topology`` with the solvated system.

    Raises
    ------
    PACKMOLRuntimeError
        When packmol fails to execute / converge.

    Notes
    -----
    Returned topologies may have larger box vectors than what would be defined
    by the target density.
    """
    _check_box_shape_shape(box_shape)

    # Compute box vectors from the solute length and requested padding
    solute_length = _max_dist_between_points(topology.get_positions())
    image_distance = solute_length + padding * 2
    box_vectors = box_shape * image_distance

    # Compute target masses of solvent
    box_volume = numpy.linalg.det(box_vectors.m) * box_vectors.u**3
    target_mass = box_volume * target_density
    solute_mass = sum(sum([atom.mass for atom in molecule.atoms]) for molecule in topology.molecules)
    solvent_mass = target_mass - solute_mass

    _check_add_positive_mass(solvent_mass)

    # Get reference data and prepare solvent molecules
    water = Molecule.from_smiles("O")
    na = Molecule.from_smiles("[Na+]")
    cl = Molecule.from_smiles("[Cl-]")
    nacl_mass = sum([atom.mass for atom in na.atoms]) + sum(
        [atom.mass for atom in cl.atoms],
    )
    water_mass = sum([atom.mass for atom in water.atoms])
    molarity_pure_water = Quantity(55.5, "mole / liter")

    # Compute the number of salt "molecules" to add from the mass and concentration
    nacl_mass_fraction = (nacl_conc * nacl_mass) / (molarity_pure_water * water_mass)
    nacl_mass_to_add = solvent_mass * nacl_mass_fraction
    nacl_to_add = (nacl_mass_to_add / nacl_mass).m_as("dimensionless").round()

    # Compute the number of water molecules to add to make up the remaining mass
    water_mass_to_add = solvent_mass - nacl_mass
    water_to_add = (water_mass_to_add / water_mass).m_as("dimensionless").round()

    # Neutralise the system by adding and removing salt
    solute_charge = sum([molecule.total_charge for molecule in topology.molecules])
    na_to_add = numpy.ceil(nacl_to_add - solute_charge.m / 2.0)
    cl_to_add = numpy.floor(nacl_to_add + solute_charge.m / 2.0)

    # Pack the box
    return pack_box(
        [water, na, cl],
        [int(water_to_add), int(na_to_add), int(cl_to_add)],
        solute=topology,
        tolerance=tolerance,
        box_vectors=box_vectors,
    )


def solvate_topology_nonwater(
    topology: Topology,
    solvent: Molecule,
    target_density: Quantity,
    padding: Quantity = Quantity(1.2, "nanometer"),
    box_shape: NDArray = RHOMBIC_DODECAHEDRON,
    tolerance: Quantity = Quantity(2.0, "angstrom"),
) -> Topology:
    """
    Solvate a topology with an arbitrary solvent.

    Parameters
    ----------
    topology
        The OpenFF Topology to solvate.
    solvent
        The OpenFF Molecule to use as the solvent.
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

    Returns
    -------
    Topology
        An OpenFF ``Topology`` with the solvated system.

    Raises
    ------
    PACKMOLRuntimeError
        When packmol fails to execute / converge.

    Notes
    -----
    Returned topologies may have larger box vectors than what would be defined
    by the target density.
    """
    _check_box_shape_shape(box_shape)

    # Compute box vectors from the solute length and requested padding
    solute_length = _max_dist_between_points(topology.get_positions())
    image_distance = solute_length + padding * 2
    box_vectors = box_shape * image_distance

    # Compute target masses of solvent
    box_volume = numpy.linalg.det(box_vectors.m) * box_vectors.u**3
    target_mass = box_volume * target_density
    solute_mass = sum(sum([atom.mass for atom in molecule.atoms]) for molecule in topology.molecules)

    solvent_mass_to_add = target_mass - solute_mass

    _check_add_positive_mass(solvent_mass_to_add)

    _solvent = deepcopy(solvent)
    solvent_mass = sum([atom.mass for atom in _solvent.atoms])

    solvent_to_add = (solvent_mass_to_add / solvent_mass).m_as("dimensionless").round()

    return pack_box(
        molecules=[solvent],
        number_of_copies=[int(solvent_to_add)],
        solute=topology,
        tolerance=tolerance,
        box_vectors=box_vectors,
        center_solute=True,
    )
