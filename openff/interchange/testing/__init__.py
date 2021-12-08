import mdtraj as md
import pytest
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.interchange.components.mdtraj import _OFFBioTop
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
        top = _OFFBioTop.from_molecules(4 * [mol])
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

    xml_ff_bo_bonds = """<?xml version='1.0' encoding='ASCII'?>
    <SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
      <Bonds version="0.3" fractional_bondorder_method="AM1-Wiberg" fractional_bondorder_interpolation="linear">
        <Bond smirks="[#6:1]~[#8:2]" id="bbo1"
            k_bondorder1="100.0 * kilocalories_per_mole/angstrom**2"
            k_bondorder2="1000.0 * kilocalories_per_mole/angstrom**2"
            length_bondorder1="1.5 * angstrom"
            length_bondorder2="1.0 * angstrom"/>
      </Bonds>
    </SMIRNOFF>
    """
