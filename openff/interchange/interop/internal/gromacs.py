"""Interfaces with GROMACS."""
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Union

import numpy as np
from openff.toolkit.topology import Topology
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.units import unit

from openff.interchange.components.base import (
    BaseAngleHandler,
    BaseBondHandler,
    BaseElectrostaticsHandler,
    BaseImproperTorsionHandler,
    BaseProperTorsionHandler,
    BasevdWHandler,
)
from openff.interchange.components.potentials import Potential
from openff.interchange.models import PotentialKey, TopologyKey

if TYPE_CHECKING:
    from openff.units.unit import Quantity

    from openff.interchange import Interchange


def _read_coordinates(file_path: Union[Path, str]) -> np.ndarray:
    def _infer_coord_precision(file_path: Union[Path, str]) -> int:
        """
        Infer decimal precision of coordinates by parsing periods in atoms lines.
        """
        with open(file_path) as file_in:
            file_in.readline()
            file_in.readline()
            atom_line = file_in.readline()
            period_indices = [i for i, x in enumerate(atom_line) if x == "."]
            spacing_between_periods = period_indices[-1] - period_indices[-2]
            precision = spacing_between_periods - 5
            return precision

    precision = _infer_coord_precision(file_path)
    coordinate_width = precision + 5
    # Column numbers in file separating x, y, z coords of each atom.
    # Default (3 decimals of precision -> 8 columns) are 20, 28, 36, 44
    coordinate_columns = [
        20,
        20 + coordinate_width,
        20 + 2 * coordinate_width,
        20 + 3 * coordinate_width,
    ]

    with open(file_path) as gro_file:
        # Throw away comment / name line
        gro_file.readline()
        n_atoms = int(gro_file.readline())

        unitless_coordinates = np.zeros((n_atoms, 3))
        for coordinate_index in range(n_atoms):
            line = gro_file.readline()
            _ = int(line[:5])  # residue_index
            _ = line[5:10]  # residue_name
            _ = line[10:15]  # atom_name
            _ = int(line[15:20])  # atom_index
            x = float(line[coordinate_columns[0] : coordinate_columns[1]])
            y = float(line[coordinate_columns[1] : coordinate_columns[2]])
            z = float(line[coordinate_columns[2] : coordinate_columns[3]])
            unitless_coordinates[coordinate_index] = np.array([x, y, z])

        coordinates = unitless_coordinates * unit.nanometer

    return coordinates


def _read_box(file_path: Union[Path, str]) -> np.ndarray:
    with open(file_path) as gro_file:
        # Throw away comment / name line
        gro_file.readline()
        n_atoms = int(gro_file.readline())

        box_line = gro_file.readlines()[n_atoms]

    parsed_box = [float(val) for val in box_line.split()]

    if len(parsed_box) == 3:
        box = parsed_box * np.eye(3) * unit.nanometer

    return box


def from_gro(file_path: Union[Path, str]) -> "Interchange":
    """Read coordinates and box information from a GROMACS GRO (.gro) file."""
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    coordinates = _read_coordinates(path)

    box = _read_box(path)

    from openff.interchange import Interchange

    interchange = Interchange()
    interchange.box = box
    interchange.positions = coordinates

    return interchange


def _get_lj_parameters(interchange: "Interchange", atom_idx: int) -> Dict:
    vdw_hander = interchange["vdW"]
    atom_key = TopologyKey(atom_indices=(atom_idx,))
    identifier = vdw_hander.key_map[atom_key]
    potential = vdw_hander.potentials[identifier]
    parameters = potential.parameters

    return parameters


# TODO: Needs to be reworked in a way that makes sane assumptions about the structure
#       of the topology while parsinng the forces and other data. This may require two
#       passes over the file, one to parse the topology and the other to parse the rest.
def from_top(top_file: Union[Path, str], gro_file: Union[Path, str]):
    """Read the contents of a GROMACS Topology (.top) file."""
    raise NotImplementedError("Internal `from_gromacs` parser temporarily unsupported.")
    from openff.interchange import Interchange

    interchange = Interchange()
    interchange.topology = Topology()
    pesudo_molecule = _SimpleMolecule()
    interchange.topology.add_molecule(pesudo_molecule)

    current_directive = None

    def _process_defaults(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 5:
            raise Exception(fields)

        nbfunc, comb_rule, gen_pairs, lj_14, coul_14 = fields

        if nbfunc == "1":
            vdw_handler = BasevdWHandler()
        elif nbfunc == "2":
            raise NotImplementedError(
                "Parsing GROMACS files with the Buckingham-6 potential is not supported",
            )

        if comb_rule == "1":
            vdw_handler.mixing_rule = "geometric"
        elif comb_rule == "2":
            vdw_handler.mixing_rule = "lorentz-berthelot"
        else:
            raise RuntimeError(f"Found bad/unsupported combination rule: '{comb_rule}'")

        # TODO: Process pairs

        electrostatics_handler = BaseElectrostaticsHandler()

        vdw_handler.scale_14 = float(lj_14)
        electrostatics_handler.scale_14 = float(coul_14)

        interchange.add_handler("vdW", vdw_handler)
        interchange.add_handler("Electrostatics", electrostatics_handler)

    def _process_atomtype(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 7:
            raise Exception

        atom_type, atomic_number, mass, charge, ptype, sigma, epsilon = fields

        potential_key = PotentialKey(id=atom_type)
        if potential_key in interchange["vdW"].potentials:
            raise RuntimeError

        potential = Potential(
            parameters={
                "sigma": float(sigma) * unit.nanometer,
                "epsilon": float(epsilon) * unit.kilojoule / unit.mole,
            },
        )

        interchange["vdW"].potentials.update({potential_key: potential})

    def _process_moleculetype(interchange: Interchange, line: str):
        from openff.toolkit.topology.molecule import Molecule

        fields = line.split()
        if len(fields) != 2:
            raise Exception

        molecule_name, nrexcl = fields

        if nrexcl != "3":
            raise Exception

        molecule = Molecule()
        molecule.name = molecule_name

        if interchange.topology is not None:
            raise Exception

        topology = molecule.to_topology()

        interchange.topology = topology

    def _process_atom(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 8:
            raise Exception

        (
            atom_number,
            atom_type,
            residue_number,
            residue_name,
            atom_name,
            cg_number,
            _charge,
            mass,
        ) = fields

        # TODO: Fix topology graph
        interchange.topology.molecules[0].add_atom(
            atomic_number=0,
            metadata={
                "residue_number": residue_number,
                "residue_name": residue_name,
            },
        )

        topology_key = TopologyKey(atom_indices=(int(atom_number) - 1,))
        potential_key = PotentialKey(id=atom_type)

        if potential_key not in interchange["vdW"].potentials:
            raise RuntimeError(
                f"Found atom type {atom_type} in an atoms directive but "
                "either did not find or failed to process an atom type of the same name "
                "in the atomtypes directive.",
            )

        charge: "Quantity" = unit.Quantity(float(_charge), units=unit.elementary_charge)

        interchange["vdW"].key_map.update({topology_key: potential_key})
        # The vdw .potentials was constructed while parsing [ atomtypes ]
        interchange["Electrostatics"].key_map.update({topology_key: potential_key})
        interchange["Electrostatics"].potentials.update(
            {potential_key: Potential(parameters={"charge": charge})},
        )

    def _process_pair(interchange: Interchange, line: str):
        pass

    def _process_bond(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 5:
            raise Exception

        if "Bonds" not in interchange.collections:
            bond_handler = BaseBondHandler()
            interchange.add_handler("Bonds", bond_handler)

        atom1, atom2, func, length, k = fields

        # Assumes 1-molecule topology
        interchange.topology.molecules[0].add_bond(
            atom1=int(atom1) - 1,
            atom2=int(atom2) - 1,
        )

        topology_key = TopologyKey(atom_indices=(int(atom1) - 1, int(atom2) - 1))
        potential_key = PotentialKey(
            id="-".join(str(i) for i in topology_key.atom_indices),
        )

        # TODO: De-depulicate identical bond parameters into "types"
        potential = Potential(
            parameters={
                "length": float(length) * unit.nanometer,
                "k": float(k) * unit.kilojoule / unit.mole / unit.nanometer**2,
            },
        )

        interchange["Bonds"].key_map.update({topology_key: potential_key})
        interchange["Bonds"].potentials.update({potential_key: potential})

    def _process_angle(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 6:
            raise Exception

        if "Angles" not in interchange.collections:
            angle_handler = BaseAngleHandler()
            interchange.add_handler("Angles", angle_handler)

        atom1, atom2, atom3, func, theta, k = fields

        topology_key = TopologyKey(
            atom_indices=(int(i) - 1 for i in [atom1, atom2, atom3]),
        )
        potential_key = PotentialKey(
            id="-".join(str(i) for i in topology_key.atom_indices),
        )

        # TODO: De-depulicate identical angle parameters into "types"
        potential = Potential(
            parameters={
                "angle": float(theta) * unit.degree,
                "k": float(k) * unit.kilojoule / unit.mole / unit.radian**2,
            },
        )

        interchange["Angles"].key_map.update({topology_key: potential_key})
        interchange["Angles"].potentials.update({potential_key: potential})

    def _process_dihedral(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 8:
            raise Exception

        if "ProperTorsions" not in interchange.collections:
            proper_handler = BaseProperTorsionHandler()
            interchange.add_handler("ProperTorsions", proper_handler)

        if "ImproperTorsions" not in interchange.collections:
            improper_handler = BaseImproperTorsionHandler()
            interchange.add_handler("ImproperTorsions", improper_handler)

        atom1, atom2, atom3, atom4, func, phase, k, periodicity = fields

        topology_key = TopologyKey(
            atom_indices=(int(i) - 1 for i in [atom1, atom2, atom3, atom4]),
            mult=0,
        )

        def ensure_unique_key(
            handler: Union[BaseProperTorsionHandler, BaseImproperTorsionHandler],
            key: TopologyKey,
        ):
            if key in handler.key_map:
                key.mult += 1
                ensure_unique_key(handler, key)

        potential_key = PotentialKey(
            id="-".join(str(i) for i in topology_key.atom_indices),
        )

        if func == "1":
            ensure_unique_key(interchange["ProperTorsions"], topology_key)
            potential_key.mult = topology_key.mult

            potential = Potential(
                parameters={
                    "phase": float(phase) * unit.degree,
                    "k": float(k) * unit.kilojoule / unit.mole,
                    "periodicity": int(periodicity) * unit.dimensionless,
                    "idivf": 1 * unit.dimensionless,
                },
            )

            interchange["ProperTorsions"].key_map.update({topology_key: potential_key})
            interchange["ProperTorsions"].potentials.update({potential_key: potential})

        elif func == "4":
            ensure_unique_key(interchange["ImproperTorsions"], topology_key)
            potential_key.mult = topology_key.mult

            potential = Potential(
                parameters={
                    "phase": float(phase) * unit.degree,
                    "k": float(k) * unit.kilojoule / unit.mole,
                    "periodicity": int(periodicity) * unit.dimensionless,
                    "idivf": 1 * unit.dimensionless,
                },
            )

            interchange["ImproperTorsions"].key_map.update(
                {topology_key: potential_key},
            )
            interchange["ImproperTorsions"].potentials.update(
                {potential_key: potential},
            )

    def _process_molecule(interchange: Interchange, line: str):
        fields = line.split()
        if len(fields) != 2:
            raise Exception

        molecule_name, n_molecules = fields

        if n_molecules != "1":
            raise NotImplementedError(
                "Only single-molecule topologies are currently supported",
            )

    def _process_system(interchange: Interchange, line: str):
        interchange.name = line

    supported_directives: Dict[str, Callable] = {
        "defaults": _process_defaults,
        "atomtypes": _process_atomtype,
        "moleculetype": _process_moleculetype,
        "atoms": _process_atom,
        "pairs": _process_pair,
        "bonds": _process_bond,
        "angles": _process_angle,
        "dihedrals": _process_dihedral,
        "molecules": _process_molecule,
        "system": _process_system,
    }

    with open(top_file) as opened_top:
        for line in opened_top:
            line = line.strip()

            if ";" in line:
                line = line[: line.index(";")].strip()
            if len(line) == 0:
                continue

            if line.startswith("#"):
                raise NotImplementedError(
                    "Parsing GROMACS files with preprocessor commands "
                    "is not yet supported. Found a line with contents "
                    f"{line}",
                )

            elif line.startswith("["):
                current_directive = line[1:-1].strip()
                continue

            if current_directive in supported_directives:
                supported_directives[current_directive](interchange, line)

            else:
                raise RuntimeError(
                    "Found bad top file or unsupported directive. "
                    f"Current directive is: {current_directive}\n"
                    f"Current line is: '{line}'",
                )

    return interchange
