"""Pytest configuration."""
import pytest
from openff.toolkit import ForceField, Molecule, Topology
from openff.toolkit.typing.engines.smirnoff.parameters import BondType, VirtualSiteType
from openff.units import Quantity, unit

from openff.interchange._tests import MoleculeWithConformer, get_test_file_path


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
def sage_with_planar_monovalent_carbonyl(sage):
    sage.get_parameter_handler("VirtualSites")
    sage["VirtualSites"].add_parameter(
        parameter=VirtualSiteType(
            smirks="[#8:1]=[#6X3+0:2]-[#6:3]",
            type="MonovalentLonePair",
            match="all_permutations",
            distance="0.5 * angstrom ** 1",
            outOfPlaneAngle=Quantity("0 * degree ** 1"),
            inPlaneAngle=Quantity("120 * degree ** 1"),
            charge_increment1="0.1 * elementary_charge ** 1",
            charge_increment2="0.1 * elementary_charge ** 1",
            charge_increment3="0.1 * elementary_charge ** 1",
        ),
    )

    return sage


@pytest.fixture()
def sage_with_trivalent_nitrogen():
    sage_210 = ForceField("openff-2.1.0.offxml")
    sage_210["Bonds"].add_parameter(
        parameter=BondType(
            smirks="[#7:3]-[#1X1:1]",
            id="b0",
            length="1.2 * angstrom",
            k="500 * angstrom**-2 * mole**-1 * kilocalorie",
        ),
    )

    sage_210.get_parameter_handler("VirtualSites")
    sage_210["VirtualSites"].add_parameter(
        parameter=VirtualSiteType(
            smirks="[#1:2][#7:1]([#1:3])[#1:4]",
            type="TrivalentLonePair",
            match="once",
            distance="0.5 * nanometer ** 1",
            outOfPlaneAngle="None",
            inPlaneAngle="None",
            charge_increment1="0.0 * elementary_charge ** 1",
            charge_increment2="0.0 * elementary_charge ** 1",
            charge_increment3="0.0 * elementary_charge ** 1",
            charge_increment4="0.0 * elementary_charge ** 1",
        ),
    )

    return sage_210


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
def tip5p() -> ForceField:
    return ForceField(
        """<?xml version="1.0" encoding="utf-8"?>
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <LibraryCharges version="0.3">
        <LibraryCharge
            name="tip5p"
            smirks="[#1:1]-[#8X2H2+0:2]-[#1:3]"
            charge1="0.*elementary_charge"
            charge2="0.*elementary_charge"
            charge3="0.*elementary_charge"/>
    </LibraryCharges>
    <vdW
        version="0.3"
        potential="Lennard-Jones-12-6"
        combining_rules="Lorentz-Berthelot"
        scale12="0.0"
        scale13="0.0"
        scale14="0.5"
        scale15="1.0"
        switch_width="0.0 * angstrom"
        cutoff="9.0 * angstrom" method="cutoff">
            <Atom
                smirks="[#1:1]-[#8X2H2+0]-[#1]"
                epsilon="0.*mole**-1*kilojoule"
                sigma="1.0 * nanometer"/>
            <Atom
                smirks="[#1]-[#8X2H2+0:1]-[#1]"
                epsilon="0.66944*mole**-1*kilojoule"
                sigma="0.312*nanometer"/>
    </vdW>
    <Constraints version="0.3">
        <Constraint
            smirks="[#1:1]-[#8X2H2+0:2]-[#1]"
            id="c-tip5p-H-O"
            distance="0.09572 * nanometer ** 1">
        </Constraint>
        <Constraint
            smirks="[#1:1]-[#8X2H2+0]-[#1:2]"
            id="c-tip5p-H-O-H"
            distance="0.15139006545 * nanometer ** 1">
        </Constraint>
    </Constraints>
    <Bonds
        version="0.4"
        potential="harmonic"
        fractional_bondorder_method="AM1-Wiberg"
        fractional_bondorder_interpolation="linear">
        <Bond
            smirks="[#1:1]-[#8X2H2+0:2]-[#1]"
            length="0.9572*angstrom"
            k="462750.4*nanometer**-2*mole**-1*kilojoule"/>
    </Bonds>
    <VirtualSites version="0.3">
        <VirtualSite
            type="DivalentLonePair"
            name="EP"
            smirks="[#1:2]-[#8X2H2+0:1]-[#1:3]"
            distance="0.70 * angstrom"
            charge_increment1="0.0*elementary_charge"
            charge_increment2="0.1205*elementary_charge"
            charge_increment3="0.1205*elementary_charge"
            sigma="10.0*angstrom"
            epsilon="0.0*kilocalories_per_mole"
            outOfPlaneAngle="54.735*degree"
            match="all_permutations" >
        </VirtualSite>
    </VirtualSites>
    <Electrostatics
        version="0.3"
        method="PME"
        scale12="0.0"
        scale13="0.0"
        scale14="0.833333"
        scale15="1.0"
        switch_width="0.0 * angstrom"
        cutoff="9.0 * angstrom"/>
</SMIRNOFF>
""",
    )


@pytest.fixture()
def gbsa_force_field() -> ForceField:
    return ForceField(
        "openff-2.0.0.offxml",
        get_test_file_path("gbsa.offxml"),
    )


@pytest.fixture()
def basic_top() -> Topology:
    topology = Molecule.from_smiles("C").to_topology()
    topology.box_vectors = unit.Quantity([5, 5, 5], unit.nanometer)
    return topology


@pytest.fixture(scope="session")
def water() -> Molecule:
    return MoleculeWithConformer.from_mapped_smiles("[H:2][O:1][H:3]")


@pytest.fixture(scope="session")
def water_dimer() -> Topology:
    return Topology.from_pdb(
        get_test_file_path("water-dimer.pdb"),
    )


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
def carbonyl_planar() -> Molecule:
    """A carbonyl group with a planar geometry."""
    # Geometry matching `planar_monovalent_carbonyl`
    return MoleculeWithConformer.from_mapped_smiles(
        "[C:3]([C:2](=[O:1])[H:7])([H:4])([H:5])[H:6]",
    )


@pytest.fixture()
def ammonia_tetrahedral() -> Molecule:
    """An ammonia molecule with a tetrahedral geometry."""
    # Sage 2.1.0; id: b87, length: 1.022553377106 angstrom
    molecule = Molecule.from_mapped_smiles("[H:2][N:1]([H:3])[H:4]")
    molecule._conformers = [
        Quantity(
            [
                [0, 0, 0.8855572013],
                [0, 0.5112766886, 0],
                [0.4427786006, -0.2556383443, 0],
                [-0.4427786006, 0.2556383443, 0],
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
