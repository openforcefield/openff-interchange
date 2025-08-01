"""Test charge assignment logging."""

import logging
import re
from collections import defaultdict

import pytest
from openff.toolkit import ForceField, Molecule, Topology
from openff.toolkit.utils import OPENEYE_AVAILABLE

from openff.interchange._tests import get_protein

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
NAGL_KEY = "openff-gnn-am1bcc-0.1.0-rc.2.pt"


def map_methods_to_atom_indices(caplog: pytest.LogCaptureFixture) -> dict[str, list[int]]:
    """
    Map partial charge assignment methods to (sorted) atom indices.
    """
    info = defaultdict(list)

    for record in caplog.records:
        # skip logged warnings from upstreams/other packages
        if record.name.startswith("openff.interchange"):
            assert record.levelname == "INFO", "Only INFO logs are expected."
        else:
            continue

        message = record.msg

        if message.startswith("Charge section LibraryCharges"):
            info["library"].append(int(message.split("atom index ")[-1]))

        elif message.startswith("Charge section ToolkitAM1BCC"):
            info[message.split(", ")[1].split(" ")[-1]].append(int(message.split("atom index ")[-1]))

        # without also pulling the virtual site - particle mapping (which is different for each engine)
        # it's hard to store more information than the orientation atoms that are affected by each
        # virtual site's charges
        elif message.startswith("Charge section VirtualSites"):
            orientation_atoms: list[int] = [
                int(value.strip()) for value in re.findall(r"\((.*?)\)", message)[0].split(",")
            ]

            for atom in orientation_atoms:
                info["orientation"].append(atom)

        elif message.startswith("Preset charges"):
            info["preset"].append(int(message.split("atom index")[-1]))

        elif message.startswith("Charge section ChargeIncrementModel"):
            if "using charge method" in message:
                info[f"chargeincrements_{message.split(',')[1].split(' ')[-1]}"].append(
                    int(message.split("atom index ")[-1]),
                )

            elif "applying charge increment" in message:
                # TODO: Store the "other" atoms?
                info["chargeincrements"].append(int(message.split("atom ")[1].split(" ")[0]))

        else:
            raise ValueError(f"Unexpected log message {message}")

    return {key: sorted(val) for key, val in info.items()}


@pytest.fixture
def sage_with_nagl_chargeincrements(sage):
    from openff.toolkit.typing.engines.smirnoff.parameters import ChargeIncrementModelHandler, ChargeIncrementType

    sage.register_parameter_handler(
        parameter_handler=ChargeIncrementModelHandler(
            version=0.3,
            partial_charge_method=NAGL_KEY,
        ),
    )

    # Add dummy "BCCs" for testing, even though this model has BCCs built into
    # the partial charge assignment method
    sage["ChargeIncrementModel"].add_parameter(
        parameter=ChargeIncrementType(
            smirks="[#6:1]-[#1:2]",
            charge_increment=[
                "0.1 elementary_charge",
                "-0.1 elementary_charge",
            ],
        ),
    )

    return sage


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
def solvent() -> Molecule:
    return Molecule.from_mapped_smiles("[H:3][C:1]([H:4])([H:5])[N:2]([H:6])[H:7]")


def ligand_in_solvent(ligand, solvent) -> Topology:
    return Topology.from_molecules(
        [
            ligand,
            solvent,
            solvent,
            solvent,
        ],
    )


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
0.xSage with just a ligand
1.xSage with a ligand in water/ions
2.xSage with a ligand in non-water solvent
3.xSage with a ligand in mixed solvent
4.xff14SB with Sage
5.xff14SB with Sage and preset charges on Protein A
6.xSage with ligand and OPC water
7.xSage with preset charges on ligand A
8.xSage with preset charges on water
9.xSage with (ligand) virtual site parameters
10.xAM1-with-custom-BCCs Sage with ligand and ions water
"""


def test_case0(caplog, sage, ligand):
    with caplog.at_level(logging.INFO):
        sage.create_interchange(ligand.to_topology())

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC
        assert info[AM1BCC_KEY] == [*range(0, 9)]


def test_case1(caplog, sage, ligand_and_water_and_ions):
    with caplog.at_level(logging.INFO):
        sage.create_interchange(ligand_and_water_and_ions)

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC
        assert info[AM1BCC_KEY] == [*range(0, 9)]

        # atoms 9 through 21 are water + ions, getting library charges
        assert info["library"] == [*range(9, 22)]


def test_case2(caplog, sage, ligand, solvent):
    topology = Topology.from_molecules([ligand, solvent, solvent, solvent])

    with caplog.at_level(logging.INFO):
        sage.create_interchange(topology)

        info = map_methods_to_atom_indices(caplog)

        # everything should get AM1-BCC charges
        assert info[AM1BCC_KEY] == [*range(0, topology.n_atoms)]


def test_case3(caplog, sage, ligand_and_water_and_ions, solvent):
    for index in range(3):
        ligand_and_water_and_ions.add_molecule(solvent)

    ligand_and_water_and_ions.molecule(0).assign_partial_charges(partial_charge_method="gasteiger")

    with caplog.at_level(logging.INFO):
        sage.create_interchange(
            ligand_and_water_and_ions,
        )

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC,
        # and also solvent molecules (starting at index 22)
        assert info[AM1BCC_KEY] == [*range(0, 9), *range(22, 22 + 3 * 7)]

        # atoms 9 through 21 are water + ions, getting library charges
        assert info["library"] == [*range(9, 22)]


@pytest.mark.slow
@pytest.mark.parametrize("preset_on_protein", [True, False])
def test_cases4_5(caplog, ligand_and_water_and_ions, preset_on_protein):
    ff = ForceField("ff14sb_off_impropers_0.0.3.offxml", "openff-2.0.0.offxml")

    complex = get_protein("MainChain_ALA_ALA").to_topology()

    for molecule in ligand_and_water_and_ions.molecules:
        complex.add_molecule(molecule)

    if preset_on_protein:
        complex.molecule(0).assign_partial_charges(partial_charge_method="zeros")

    with caplog.at_level(logging.INFO):
        if preset_on_protein:
            ff.create_interchange(complex, charge_from_molecules=[complex.molecule(0)])
        else:
            ff.create_interchange(complex)

        info = map_methods_to_atom_indices(caplog)

        assert info[AM1BCC_KEY] == [*range(complex.molecule(0).n_atoms, complex.molecule(0).n_atoms + 9)]

        if preset_on_protein:
            # protein gets preset charges
            assert info["preset"] == [*range(0, complex.molecule(0).n_atoms)]

            # everything after the protein and ligand should get library charges
            assert info["library"] == [
                *range(complex.molecule(0).n_atoms + 9, complex.n_atoms),
            ]
        else:
            # the protein and everything after the ligand should get library charges
            assert info["library"] == [
                *range(0, complex.molecule(0).n_atoms),
                *range(complex.molecule(0).n_atoms + 9, complex.n_atoms),
            ]


def test_case6(caplog, ligand, water):
    force_field = ForceField("openff-2.0.0.offxml", "opc.offxml")

    topology = Topology.from_molecules([ligand, water, water, water])

    with caplog.at_level(logging.INFO):
        force_field.create_interchange(topology)

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting AM1-BCC charges
        assert info[AM1BCC_KEY] == [*range(0, 9)]

        # atoms 9 through 17 are water atoms, getting library charges
        assert info["library"] == [*range(9, 18)]

        # particles 18 through 20 are water virtual sites, but the current logging strategy
        # makes it hard to match these up (and the particle indices are different OpenMM/GROMACS/etc)

        # can still check that orientation atoms are subject to virtual site
        # charge increments (even if the increment is +0.0 e)
        assert info["orientation"] == [*range(9, 18)]


def test_case7(caplog, sage, ligand_and_water_and_ions):
    ligand_and_water_and_ions.molecule(0).assign_partial_charges(partial_charge_method="gasteiger")

    with caplog.at_level(logging.INFO):
        sage.create_interchange(
            ligand_and_water_and_ions,
            charge_from_molecules=[ligand_and_water_and_ions.molecule(0)],
        )

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are ethanol, getting preset charges
        assert info["preset"] == [*range(0, 9)]

        # atoms 9 through 21 are water + ions, getting library charges
        assert info["library"] == [*range(9, 22)]


def test_case8(caplog, sage, water_and_ions):
    water_and_ions.molecule(0).assign_partial_charges(partial_charge_method="gasteiger")

    with caplog.at_level(logging.INFO):
        sage.create_interchange(
            water_and_ions,
            charge_from_molecules=[water_and_ions.molecule(0)],
        )

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 8 are water, getting preset charges
        assert info["preset"] == [*range(0, 9)]

        # atoms 9 through 12 are ions, getting library charges
        assert info["library"] == [*range(9, 13)]


def test_case9(caplog, sage_with_bond_charge):
    with caplog.at_level(logging.INFO):
        sage_with_bond_charge.create_interchange(
            Molecule.from_mapped_smiles(
                "[H:3][C:1]([H:4])([H:5])[Cl:2]",
            ).to_topology(),
        )

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 5 are ligand, getting AM1-BCC charges
        assert info[AM1BCC_KEY] == [*range(0, 5)]

        # atoms 0 and 1 are the orientation atoms of the sigma hole virtual site
        assert info["orientation"] == [0, 1]


def test_case10(caplog, sage_with_nagl_chargeincrements, ligand):
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
    from openff.toolkit.utils.toolkit_registry import ToolkitRegistry, toolkit_registry_manager

    pytest.importorskip("openff.nagl")
    pytest.importorskip("rdkit")

    with caplog.at_level(logging.INFO):
        with toolkit_registry_manager(
            toolkit_registry=ToolkitRegistry(
                toolkit_precedence=[NAGLToolkitWrapper, RDKitToolkitWrapper],
            ),
        ):
            sage_with_nagl_chargeincrements.create_interchange(ligand.to_topology())

        info = map_methods_to_atom_indices(caplog)

        # atoms 0 through 5 are ligand, getting NAGL charges
        assert info[f"chargeincrements_{NAGL_KEY}"] == [*range(0, 9)]

        # TODO: These are logged symmetrically (i.e. hydrogens are listed)
        # even though the charges appear to be correct, assert should
        # simply by == [0, 1] since the hydrogens shouldn't be caught
        assert 0 in info["chargeincrements"]
        assert 1 in info["chargeincrements"]

        # the standard AM1-BCC should not have ran
        assert AM1BCC_KEY not in info
