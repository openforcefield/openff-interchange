import mdtraj as md
import pytest
from openff.toolkit.tests.utils import get_data_file_path
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange.testing.utils import _top_from_smiles
from openff.interchange.utils import get_test_file_path


class _BaseTest:
    @pytest.fixture(autouse=True)
    def _initdir(self, tmpdir):
        tmpdir.chdir()

    # TODO: group fixtures up as dicts, i.e. argon['forcefield'], argon['topology'], ...
    @pytest.fixture()
    def argon_ff(self):
        """Fixture that loads an SMIRNOFF XML for argon."""
        return ForceField(get_test_file_path("argon.offxml"))

    @pytest.fixture()
    def argon_top(self):
        """Fixture that builds a simple arogon topology."""
        return _top_from_smiles("[#18]")

    @pytest.fixture()
    def ammonia_ff(self):
        """Fixture that loads an SMIRNOFF XML for ammonia."""
        return ForceField(get_test_file_path("ammonia.offxml"))

    @pytest.fixture()
    def ammonia_top(self):
        """Fixture that builds a simple ammonia topology."""
        mol = Molecule.from_smiles("N")
        top = Topology.from_molecules(4 * [mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())
        return top

    @pytest.fixture()
    def ethanol_top(self):
        """Fixture that builds a simple four ethanol topology."""
        return _top_from_smiles("CCO", n_molecules=4)

    @pytest.fixture()
    def parsley(self):
        return ForceField("openff-1.0.0.offxml")

    @pytest.fixture()
    def parsley_unconstrained(self):
        return ForceField("openff_unconstrained-1.0.0.offxml")

    @pytest.fixture()
    def mainchain_ala(self):
        molecule = Molecule.from_file(get_data_file_path("proteins/MainChain_ALA.sdf"))
        molecule._add_default_hierarchy_schemes()
        molecule.perceive_residues()
        molecule.perceive_hierarchy()

        return molecule

    @pytest.fixture()
    def mainchain_arg(self):
        molecule = Molecule.from_file(get_data_file_path("proteins/MainChain_ARG.sdf"))
        molecule._add_default_hierarchy_schemes()
        molecule.perceive_residues()
        molecule.perceive_hierarchy()

        return molecule

    @pytest.fixture()
    def two_peptides(self, mainchain_ala, mainchain_arg):
        return Topology.from_molecules([mainchain_ala, mainchain_arg])
