import pytest
from openff.toolkit import Molecule
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.interchange._tests import _BaseTest
from openff.interchange.exceptions import UnsupportedCutoffMethodError


@skip_if_missing("openmm")
class TestUnsupportedCases(_BaseTest):
    def test_ljpme_nonperiodic(self, sage):
        interchange = sage.create_interchange(Molecule.from_smiles("CC").to_topology())

        interchange["vdW"].nonperiodic_method = "pme"

        with pytest.raises(
            UnsupportedCutoffMethodError,
            match="not valid for non-periodic systems",
        ):
            interchange.to_openmm(combine_nonbonded_forces=False)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_reaction_field(self, sage, periodic):
        interchange = sage.create_interchange(Molecule.from_smiles("CC").to_topology())

        if periodic:
            interchange.box = [4, 4, 4] * unit.nanometer
            interchange["Electrostatics"].periodic_potential = "reaction-field"
        else:
            interchange["Electrostatics"].nonperiodic_potential = "reaction-field"

        with pytest.raises(
            UnsupportedCutoffMethodError,
            match="Reaction field electrostatics not supported. ",
        ):
            interchange.to_openmm(combine_nonbonded_forces=False)


@skip_if_missing("openmm")
class TestEwaldSettings:
    def test_set_ewald_tolerance(self, sage, basic_top):
        import openmm

        system = sage.create_interchange(basic_top).to_openmm(ewald_tolerance=1.234e-5)

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getEwaldErrorTolerance() == 1.234e-5
                break
        else:
            pytest.fail("Found no `NonbondedForce`")
