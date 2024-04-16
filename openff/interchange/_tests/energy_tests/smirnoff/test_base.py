from openff.toolkit import Quantity
from openff.utilities.testing import skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, needs_gmx
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies


@needs_gmx
@skip_if_missing("openmm")
def test_issue_908(sage_unconstrained):
    molecule = MoleculeWithConformer.from_smiles("ClCCl")
    topology = molecule.to_topology()
    topology.box_vectors = Quantity([4, 4, 4], "nanometer")

    state1 = sage_unconstrained.create_interchange(topology)

    with open("test.json", "w") as f:
        f.write(state1.json())

    state2 = Interchange.parse_file("test.json")

    get_gromacs_energies(state1).compare(get_gromacs_energies(state2))
    get_openmm_energies(
        state1,
        combine_nonbonded_forces=False,
    ).compare(get_openmm_energies(state2, combine_nonbonded_forces=False))
