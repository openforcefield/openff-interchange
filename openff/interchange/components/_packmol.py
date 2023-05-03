"""
A wrapper around PACKMOL. Taken from OpenFF Evaluator v0.4.3.
"""
import os
import shutil
import subprocess
import tempfile
from distutils.spawn import find_executable
from typing import Optional

import numpy as np
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
    topology_to_solvate: Optional[Topology],
    box_aspect_ratio: Optional[Quantity],
    box_size: Optional[Quantity],
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
    topology_to_solvate: Topology, optional
        The OpenFF Topology to be solvated.
    box_size : openff.units.Quantity, optional
        The size of the box to generate in units compatible with angstroms.
        If `None`, `mass_density` must be provided.
    mass_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
         If `None`, `box_size` must be provided.
    box_aspect_ratio: list of float, optional
        The aspect ratio of the simulation box, used in conjunction with
        the `mass_density` parameter.

    """
    if box_size is None and mass_density is None:
        raise PACKMOLValueError(
            "Either a `box_size` or `mass_density` must be specified.",
        )

    if box_size is not None and len(box_size) != 3:
        raise PACKMOLValueError(
            "`box_size` must be a openff.units.unit.Quantity wrapped list of length 3",
        )

    if box_aspect_ratio is not None:
        assert len(box_aspect_ratio) == 3
        assert all(x > 0.0 for x in box_aspect_ratio)

    if len(molecules) != len(number_of_copies):
        raise PACKMOLValueError(
            "The length of `molecules` and `number_of_copies` must be identical.",
        )

    if topology_to_solvate is not None:
        assert topology_to_solvate.get_positions() is not None


def _approximate_box_size_by_density(
    molecules: list[Molecule],
    n_copies: list[int],
    mass_density: Quantity,
    box_aspect_ratio: list[float],
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
    box_aspect_ratio: List of float
        The aspect ratio of the simulation box, used in conjunction with
        the `mass_density` parameter.

    Returns
    -------
    openff.units.Quantity
        A list of the three box lengths in units compatible with angstroms.

    """
    total_mass = 0.0 * unit.dalton
    for molecule, number in zip(molecules, n_copies):
        for atom in molecule.atoms:
            total_mass += number * atom.mass
    volume = total_mass / mass_density

    box_length = volume ** (1.0 / 3.0)
    box_length_angstrom = box_length.to(unit.angstrom).magnitude

    aspect_ratio_normalizer = (
        box_aspect_ratio[0] * box_aspect_ratio[1] * box_aspect_ratio[2]
    ) ** (1.0 / 3.0)

    box_size = [
        box_length_angstrom * box_aspect_ratio[0],
        box_length_angstrom * box_aspect_ratio[1],
        box_length_angstrom * box_aspect_ratio[2],
    ] * unit.angstrom

    box_size /= aspect_ratio_normalizer

    return box_size


def _build_input_file(
    molecule_file_names: list[str],
    molecule_counts: list[int],
    structure_to_solvate: Optional[str],
    center_solute: bool,
    box_size: Quantity,
    tolerance: Quantity,
    output_file_name: str,
) -> str:
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
    center_solute: bool
        If `True`, the structure to solvate will be centered in the
        simulation box.
    box_size: openff.units.Quantity
        The lengths of each box vector. This is the box size of the simulation
        box; the packmol box will be shrunk by the tolerance.
    tolerance: openff.units.Quantity
        The packmol convergence tolerance.
    output_file_name: str
        The path to save the packed pdb to.

    Returns
    -------
    str
        The path to the input file.

    """
    box_size = (box_size - tolerance).m_as(unit.angstrom)
    tolerance = tolerance.m_as(unit.angstrom)

    # Add the global header options.
    input_lines = [
        f"tolerance {tolerance:f}",
        "filetype pdb",
        f"output {output_file_name}",
        "",
    ]

    # Add the section of the molecule to solvate if provided.
    if structure_to_solvate is not None:
        solute_position = [0.0] * 3

        if center_solute:
            solute_position = [box_size[i] / 2.0 for i in range(3)]

        input_lines.extend(
            [
                f"structure {structure_to_solvate}",
                "  number 1",
                "  fixed "
                f"{solute_position[0]} "
                f"{solute_position[1]} "
                f"{solute_position[2]} "
                "0. 0. 0.",
                "centerofmass" if center_solute else "",
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

    return packmol_file_name


@requires_package("mdtraj")
def pack_box(
    molecules: list[Molecule],
    number_of_copies: list[int],
    topology_to_solvate: Optional[Topology] = None,
    center_solute: bool = True,
    tolerance: Quantity = 2.0 * unit.angstrom,
    box_size: Optional[Quantity] = None,
    mass_density: Optional[Quantity] = None,
    box_aspect_ratio: Optional[list[float]] = None,
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
    topology_to_solvate: Topology, optional
        The OpenFF Topology to be solvated.
    center_solute: bool
        If ``True``, the structure to solvate will be placed in the center of
        the simulation box. This option is only applied when
        ``structure_to_solvate`` is set.
    tolerance : openff.units.Quantity
        The minimum spacing between molecules during packing in units of
        distance. The default is large so that added waters do not disrupt the
        structure of proteins; when constructing a mixture of small molecules,
        values as small as 0.05 nm will converge faster and can still produce
        stable simulations after energy minimisation.
    box_size : openff.units.Quantity, optional
        The size of the box to generate in units compatible with angstroms.
        If ``None``, ``mass_density`` must be provided.
    mass_density : openff.units.Quantity, optional
        Target mass density for final system with units compatible with g / mL.
        If ``None``, ``box_size`` must be provided.
    box_aspect_ratio: list of float, optional
        The aspect ratio of the simulation box, used in conjunction with
        the ``mass_density`` parameter. If none, an isotropic ratio (i.e.
        ``[1.0, 1.0, 1.0]``) is used.
    verbose : bool
        If ``True``, verbose output is written.
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
    if mass_density is not None and box_aspect_ratio is None:
        box_aspect_ratio = [1.0, 1.0, 1.0]

    # Make sure packmol can be found.
    packmol_path = _find_packmol()

    if packmol_path is None:
        raise OSError("Packmol not found, cannot run pack_box()")

    # Validate the inputs.
    _validate_inputs(
        molecules,
        number_of_copies,
        topology_to_solvate,
        box_aspect_ratio,
        box_size,
        mass_density,
    )

    # Estimate the box_size from mass density if one is not provided.
    if box_size is None:
        box_size = _approximate_box_size_by_density(
            molecules,
            number_of_copies,
            mass_density,
            box_aspect_ratio,  # type: ignore[arg-type]
        )

    # Set up the directory to create the working files in.
    temporary_directory = False

    if working_directory is None:
        working_directory = tempfile.mkdtemp()
        temporary_directory = True

    if len(working_directory) > 0:
        os.makedirs(working_directory, exist_ok=True)

    with temporary_cd(working_directory):
        # Write out the topology to solvate so that packmol can read it
        if topology_to_solvate is not None:
            structure_to_solvate = "solvate.pdb"
            topology_to_solvate.to_file(
                structure_to_solvate,
                file_format="PDB",
            )
        else:
            structure_to_solvate = None
            topology_to_solvate = Topology()

        # Create PDB files for all of the molecules.
        pdb_file_names = []

        for index, molecule in enumerate(molecules):
            # Make a copy of the molecule so we don't change input
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

        # Generate the input file.
        output_file_name = "packmol_output.pdb"

        input_file_path = _build_input_file(
            pdb_file_names,
            number_of_copies,
            structure_to_solvate,
            center_solute,
            box_size,
            tolerance,
            output_file_name,
        )

        with open(input_file_path) as file_handle:
            result = subprocess.check_output(
                packmol_path,
                stdin=file_handle,
                stderr=subprocess.STDOUT,
            ).decode("utf-8")

            packmol_succeeded = result.find("Success!") > 0

        if not retain_working_files:
            os.unlink(input_file_path)

            for file_path in pdb_file_names:
                os.unlink(file_path)

        if not packmol_succeeded:
            if os.path.isfile(output_file_name):
                os.unlink(output_file_name)

            if temporary_directory and not retain_working_files:
                shutil.rmtree(working_directory)

            raise PACKMOLRuntimeError(result)

        # Construct the output topology
        added_molecules = []
        for mol, n in zip(molecules, number_of_copies):
            added_molecules.extend([mol] * n)

        import mdtraj

        topology = topology_to_solvate + Topology.from_molecules(added_molecules)
        topology.set_positions(
            mdtraj.load(output_file_name).xyz.reshape(-1, 3) * unit.nanometer,
        )
        topology.box_vectors = box_size * UNIT_CUBE

        if not retain_working_files:
            os.unlink(output_file_name)

    if temporary_directory and not retain_working_files:
        shutil.rmtree(working_directory)

    return topology
