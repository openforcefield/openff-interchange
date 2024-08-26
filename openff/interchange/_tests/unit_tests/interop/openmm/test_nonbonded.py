import pytest
from openff.toolkit import Molecule, unit
from openff.utilities.testing import skip_if_missing

from openff.interchange.exceptions import (
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)


@skip_if_missing("openmm")
class TestUnsupportedCases:
    def test_ljpme_nonperiodic(self, sage):
        interchange = sage.create_interchange(Molecule.from_smiles("CC").to_topology())

        interchange["vdW"].nonperiodic_method = "pme"

        with pytest.raises(
            UnsupportedCutoffMethodError,
            match="not valid for non-periodic systems",
        ):
            interchange.to_openmm(combine_nonbonded_forces=False)

    @pytest.mark.parametrize("periodic", [True, False])
    @pytest.mark.parametrize("combine", [True, False])
    def test_hard_cutoff(self, sage, periodic, combine):
        interchange = sage.create_interchange(Molecule.from_smiles("CC").to_topology())

        if periodic:
            interchange.box = [4, 4, 4] * unit.nanometer
            interchange["Electrostatics"].periodic_potential = "cutoff"
        else:
            interchange["Electrostatics"].nonperiodic_potential = "cutoff"

        with pytest.raises(
            UnsupportedCutoffMethodError,
            match="does not support.*Consider using",
        ):
            interchange.to_openmm(combine_nonbonded_forces=combine)


@skip_if_missing("openmm")
class TestCutoffElectrostatics:
    @pytest.mark.parametrize("periodic", [True, False])
    @pytest.mark.parametrize("combine", [True, False])
    def test_export_reaction_field_electrostatics(
        self,
        sage,
        basic_top,
        periodic,
        combine,
    ):
        import openmm
        import openmm.unit

        out = sage.create_interchange(basic_top)

        if periodic:
            out["Electrostatics"].periodic_potential = "reaction-field"
        else:
            out["Electrostatics"].nonperiodic_potential = "reaction-field"
            out.box = None

        assert (out.box is not None) == periodic

        out["Electrostatics"].cutoff = out["vdW"].cutoff

        if combine and not periodic:
            with pytest.raises(
                UnsupportedCutoffMethodError,
                match="Combination of",
            ):
                system = out.to_openmm(combine_nonbonded_forces=combine)
            return

        system = out.to_openmm(combine_nonbonded_forces=combine)

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert 0.9 == force.getCutoffDistance() / openmm.unit.nanometer

                if periodic:
                    expected_force = openmm.NonbondedForce.CutoffPeriodic
                else:
                    expected_force = openmm.NonbondedForce.CutoffNonPeriodic

                assert expected_force == force.getNonbondedMethod()

                del force

                break
        else:
            pytest.fail("Found no `NonbondedForce`")

    def test_vdw_electrostatics_cutoff_mismatch(
        self,
        sage,
        basic_top,
    ):
        import random

        out = sage.create_interchange(basic_top)

        out["Electrostatics"].periodic_potential = "reaction-field"
        out["Electrostatics"].cutoff = out["vdW"].cutoff

        for key in ("vdW", "Electrostatics"):
            out[key].cutoff *= random.random()

        with pytest.raises(
            UnsupportedExportError,
            match="cutoffs must match",
        ):
            out.to_openmm(combine_nonbonded_forces=True)


@skip_if_missing("openmm")
class TestEwaldSettings:
    @pytest.mark.parametrize("lj_method", ["cutoff", "pme"])
    def test_set_ewald_tolerance(self, sage, basic_top, lj_method):
        import openmm

        if lj_method == "pme":
            sage["vdW"].periodic_method = "Ewald3D"

        system = sage.create_interchange(basic_top).to_openmm(ewald_tolerance=1.234e-5)

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                if lj_method == "pme":
                    assert force.getNonbondedMethod() == openmm.NonbondedForce.LJPME
                elif lj_method == "cutoff":
                    assert force.getNonbondedMethod() == openmm.NonbondedForce.PME

                assert force.getEwaldErrorTolerance() == 1.234e-5

                break
        else:
            pytest.fail("Found no `NonbondedForce`")
