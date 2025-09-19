import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology

from openff.interchange.drivers.amber import AmberError, get_amber_energies


def test_error_bad_vdw():
    x = ForceField("openff-2.0.0.offxml").create_interchange(
        Topology.from_molecules(
            [
                Molecule.from_smiles("[Na+]"),
                Molecule.from_smiles("[Cl-]"),
            ],
        ),
    )

    x.box = [4, 4, 4]

    x.positions = Quantity(
        [
            [0, 0, 0],
            [0.01, 0, 0],
        ],
        "angstrom",
    )

    with pytest.raises(
        AmberError,
        match=r"Found bad energy value ' *************' associated with energy type 'VDWAALS'",
    ):
        get_amber_energies(x)
