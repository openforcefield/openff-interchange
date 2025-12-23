from openff.toolkit import Quantity
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_gmx
from openff.interchange.drivers.gromacs import get_gromacs_energies
from openff.interchange.drivers.openmm import get_openmm_energies


@skip_if_missing("openmm")
@needs_gmx
def test_group_impropers(cb8_host, no_charges):
    out = Interchange.from_smirnoff(no_charges, [cb8_host], box=[4, 4, 4])

    openmm_torsions = get_openmm_energies(out)["Torsion"]
    gromacs_torsions = get_gromacs_energies(out)["Torsion"]

    assert abs(openmm_torsions - gromacs_torsions).m_as("kilojoule_per_mole") < 1e-3


@needs_gmx
def test_14_in_detailed_report(sage):
    """Ensure that 1-4 interactions are included in the detailed report and non-zero."""
    octanol = MoleculeWithConformer.from_smiles("CCCCCCCC")
    topology = octanol.to_topology()
    topology.box_vectors = Quantity([3, 3, 3], "nanometer")

    report = get_gromacs_energies(sage.create_interchange(topology), detailed=True)

    assert report["vdW 1-4"].m != 0.0
    assert report["Electrostatics 1-4"].m != 0.0


@needs_gmx
def test_key_order(methane_dimer):
    """Ensure that the keys in the report are in a consistent order."""
    report = get_gromacs_energies(methane_dimer, detailed=True)

    expected_keys = [
        "Bond",
        "Angle",
        "Torsion",
        "RBTorsion",
        "vdW",
        "vdW 1-4",
        "Electrostatics",
        "Electrostatics 1-4",
    ]

    assert [*report.energies.keys()] == [key for key in expected_keys if key in report.energies]
