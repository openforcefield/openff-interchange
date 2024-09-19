"""Test charge assignment logging."""

import logging
from collections import defaultdict

import pytest
from openff.toolkit import Molecule, Topology
from openff.toolkit.utils import OPENEYE_AVAILABLE

"""
Hierarchy is
1. Match molceules with preset charges
2. Match chemical environment against library charges
3. Match chemical environment against charge increments
4. Run AM1-BCC (or a variant) on remaining molecules

Test cases
----------

0. Sage with a ligand in vacuum
* Ligand gets AM1-BCC

1. Sage with a ligand in water/ions
* Water gets library charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

2. Sage with a ligand in non-water solvent
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

2. Sage with a ligand in mixed solvent
* Water gets library charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

4. ff14SB with Sage
* Protein gets library charges
* Water gets library charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

5. ff14SB with Sage and preset charges on Protein A
* Protein A gets preset charges
* Other proteins get library charges
* Water gets library charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

6. Sage with ligand and OPC water
* Water gets library charges
* Water virtual sites get ??
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

7. Sage with preset charges on ligand A
* Water gets library charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligand A gets preset charges
* Other ligands get AM1-BCC

8. Sage with preset charges on water
* Water gets preset charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligands get AM1-BCC

9. Sage with (ligand) virtual site parameters
* Water gets preset charges
* Ions get library charges
* Non-water solvent gets AM1-BCC
* Ligand heavy atoms get AM1-BCC and charge increments
* Virtual sites get charge increments

10. AM1-with-custom-BCCs Sage with ligand and ions water
* Water gets library charges
* Ions get library charges
* Ligand gets charge increments

Other details
* Specifics of charge method (AM1-BCC, AM1-BCC-ELF10, AM1-BCC via NAGL)
* Molecules with preset charges can be similar but not exact enough
* Preset charges and virtual sites is underspecified and cannot be tested
"""

# TODO: Handle whether or not NAGL/graph charges are applied
AM1BCC_KEY = "am1bccelf10" if OPENEYE_AVAILABLE else "am1bcc"


def map_methods_to_atom_indices(caplog: pytest.LogCaptureFixture) -> dict[str, list[int]]:
    """
    Map partial charge assignment methods to atom indices.
    """
    info = defaultdict(list)

    for record in caplog.records:
        assert record.levelname == "INFO", "Only INFO logs are expected."

        message = record.msg

        if message.startswith("Charge section LibraryCharges"):
            info["library"].append(int(message.split("atom index ")[-1]))

        elif message.startswith("Charge section ToolkitAM1BCC"):
            info[caplog.records[0].msg.split(", ")[1].split(" ")[-1]].append(int(message.split("atom index ")[-1]))

        elif message.startswith("Preset charges"):
            info["preset"].append(int(message.split("atom index")[-1]))

        else:
            raise ValueError(f"Unexpected log message {message}")

    return info


def check_method(
    atom_indices: tuple[int],
    method: str,
):
    """
    Check logs that a given set of atom indices was assigned by a given partial charge assignment method.
    """
    pass


@pytest.fixture
def ions() -> Topology:
    return Topology.from_molecules(
        [
            Molecule.from_smiles(smiles)
            for smiles in [
                "[Na+]",
                "[Cl-]",
                "[Na+]",
                "[Cl-]",
            ]
        ],
    )


@pytest.fixture
def ligand() -> Molecule:
    return Molecule.from_mapped_smiles("[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]")


@pytest.fixture
def water_and_ions(water, ions) -> Topology:
    topology = Topology.from_molecules(3 * [water])
    for molecule in ions.molecules:
        topology.add_molecule(molecule)

    return topology


@pytest.fixture
def ligand_and_water_and_ions(ligand, water_and_ions) -> Topology:
    topology = ligand.to_topology()

    for molecule in water_and_ions.molecules:
        topology.add_molecule(molecule)

    return topology


"""
1. Sage with a ligand in water/ions
2. Sage with a ligand in non-water solvent
3. Sage with a ligand in mixed solvent
4. ff14SB with Sage
5. ff14SB with Sage and preset charges on Protein A
6. Sage with ligand and OPC water
7. Sage with preset charges on ligand A
8. Sage with preset charges on water
9. Sage with (ligand) virtual site parameters
10. AM1-with-custom-BCCs Sage with ligand and ions water
"""


def test_case0(caplog, sage, ligand):
    with caplog.at_level(logging.INFO):
        sage.create_interchange(ligand.to_topology())

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC
        assert sorted(info[AM1BCC_KEY]) == [*range(0, 9)]


def test_case1(caplog, sage, ligand_and_water_and_ions):
    with caplog.at_level(logging.INFO):
        sage.create_interchange(ligand_and_water_and_ions)

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC
        assert sorted(info[AM1BCC_KEY]) == [*range(0, 9)]

        # atoms 9 through 21 are water + ions, getting library charges
        assert sorted(info["library"]) == [*range(9, 22)]


def test_case7(caplog, sage, ligand_and_water_and_ions):
    ligand_and_water_and_ions.molecule(0).assign_partial_charges(partial_charge_method="gasteiger")

    with caplog.at_level(logging.INFO):
        sage.create_interchange(
            ligand_and_water_and_ions,
            charge_from_molecules=[ligand_and_water_and_ions.molecule(0)],
        )

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting preset charges
        assert sorted(info["preset"]) == [*range(0, 9)]

        # atoms 9 through 21 are water + ions, getting library charges
        assert sorted(info["library"]) == [*range(9, 22)]
