"""Pytest configuration."""
import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import BondType, VirtualSiteType
from openff.units import Quantity, unit

from openff.interchange._tests import get_test_file_path


@pytest.fixture()
def sage():
    return ForceField("openff-2.0.0.offxml")


@pytest.fixture()
def sage_unconstrained():
    return ForceField("openff_unconstrained-2.0.0.offxml")


@pytest.fixture()
def sage_with_bond_charge(sage):
    sage["Bonds"].add_parameter(
        parameter=BondType(
            smirks="[#6:2]-[#17X1:1]",
            id="b0",
            length="1.6 * angstrom",
            k="500 * angstrom**-2 * mole**-1 * kilocalorie",
        ),
    )

    sage.get_parameter_handler("VirtualSites")
    sage["VirtualSites"].add_parameter(
        parameter=VirtualSiteType(
            smirks="[#6:2]-[#17X1:1]",
            type="BondCharge",
            match="all_permutations",
            distance="0.8 * angstrom ** 1",
            charge_increment1="0.0 * elementary_charge ** 1",
            charge_increment2="0.0 * elementary_charge ** 1",
        ),
    )

    return sage


@pytest.fixture()
def _simple_force_field():
    # TODO: Create a minimal force field for faster tests
    pass


@pytest.fixture()
def tip3p() -> ForceField:
    return ForceField("tip3p.offxml")


@pytest.fixture()
def tip4p() -> ForceField:
    return ForceField("tip4p_fb.offxml")


@pytest.fixture()
def gbsa_force_field() -> ForceField:
    return ForceField(
        "openff-2.0.0.offxml",
        get_test_file_path("gbsa.offxml"),
    )


@pytest.fixture(scope="session")
def water() -> Molecule:
    molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    molecule.generate_conformers(n_conformers=1)
    return molecule


@pytest.fixture(scope="session")
def water_tip4p() -> Molecule:
    """Water minimized to TIP4P-FB geometry."""
    molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    molecule._conformers = [
        Quantity(
            [
                [0.0, 0.0, 0.0],
                [-0.7569503, -0.5858823, 0.0],
                [0.7569503, -0.5858823, 0.0],
            ],
            unit.angstrom,
        ),
    ]
    return molecule


@pytest.fixture()
def ethanol() -> Molecule:
    ethanol = Molecule()

    ethanol.add_atom(6, 0, False)
    ethanol.add_atom(6, 0, False)
    ethanol.add_atom(8, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_bond(0, 1, 1, False, fractional_bond_order=1.33)
    ethanol.add_bond(1, 2, 1, False, fractional_bond_order=1.23)
    ethanol.add_bond(0, 3, 1, False, fractional_bond_order=1)
    ethanol.add_bond(0, 4, 1, False, fractional_bond_order=1)
    ethanol.add_bond(0, 5, 1, False, fractional_bond_order=1)
    ethanol.add_bond(1, 6, 1, False, fractional_bond_order=1)
    ethanol.add_bond(1, 7, 1, False, fractional_bond_order=1)
    ethanol.add_bond(2, 8, 1, False, fractional_bond_order=1)

    ethanol.partial_charges = Quantity(
        [-0.4, -0.3, -0.2, -0.1, 0.00001, 0.1, 0.2, 0.3, 0.4],
        unit.elementary_charge,
    )

    return ethanol


@pytest.fixture()
def reversed_ethanol() -> Molecule:
    ethanol = Molecule()

    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(1, 0, False)
    ethanol.add_atom(8, 0, False)
    ethanol.add_atom(6, 0, False)
    ethanol.add_atom(6, 0, False)
    ethanol.add_bond(8, 7, 1, False, fractional_bond_order=1.33)
    ethanol.add_bond(7, 6, 1, False, fractional_bond_order=1.23)
    ethanol.add_bond(8, 5, 1, False, fractional_bond_order=1)
    ethanol.add_bond(8, 4, 1, False, fractional_bond_order=1)
    ethanol.add_bond(8, 3, 1, False, fractional_bond_order=1)
    ethanol.add_bond(7, 2, 1, False, fractional_bond_order=1)
    ethanol.add_bond(7, 1, 1, False, fractional_bond_order=1)
    ethanol.add_bond(6, 0, 1, False, fractional_bond_order=1)

    ethanol.partial_charges = Quantity(
        [0.4, 0.3, 0.2, 0.1, 0.00001, -0.1, -0.2, -0.3, -0.4],
        unit.elementary_charge,
    )

    return ethanol


@pytest.fixture()
def cyclohexane() -> Molecule:
    cyclohexane = Molecule()

    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(6, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_atom(1, 0, False)
    cyclohexane.add_bond(0, 1, 1, False)
    cyclohexane.add_bond(1, 2, 1, False)
    cyclohexane.add_bond(2, 3, 1, False)
    cyclohexane.add_bond(3, 4, 1, False)
    cyclohexane.add_bond(4, 5, 1, False)
    cyclohexane.add_bond(5, 0, 1, False)
    cyclohexane.add_bond(0, 6, 1, False)
    cyclohexane.add_bond(0, 7, 1, False)
    cyclohexane.add_bond(1, 8, 1, False)
    cyclohexane.add_bond(1, 9, 1, False)
    cyclohexane.add_bond(2, 10, 1, False)
    cyclohexane.add_bond(2, 11, 1, False)
    cyclohexane.add_bond(3, 12, 1, False)
    cyclohexane.add_bond(3, 13, 1, False)
    cyclohexane.add_bond(4, 14, 1, False)
    cyclohexane.add_bond(4, 15, 1, False)
    cyclohexane.add_bond(5, 16, 1, False)
    cyclohexane.add_bond(5, 17, 1, False)

    return cyclohexane
