import pytest

from openff.interchange._tests import _BaseTest
from openff.interchange.components.mdconfig import (
    MDConfig,
    get_intermol_defaults,
    get_smirnoff_defaults,
)
from openff.interchange.constants import _PME


def parse_mdp(file: str) -> dict[str, str]:
    """Naively parse an MDP file into a dict structure."""
    options = dict()

    with open(file) as f:
        for line in f.readlines():
            split = [token.replace("-", "") for token in line.split()]

            if len(split) == 0:
                continue

            elif len(split) == 3:
                assert split[1] == "="

                options[split[0].lower()] = split[2].lower()

            else:
                raise Exception

    return options


class TestMDConfigFromInterchange(_BaseTest):
    @pytest.mark.parametrize("switch", [True, False])
    def test_from_interchange(self, sage, basic_top, switch):
        from openff.units import unit

        from openff.interchange import Interchange

        if not switch:
            sage["vdW"].switch_width = 0.0 * unit.nanometer

        interchange = Interchange.from_smirnoff(sage, basic_top)
        config = MDConfig.from_interchange(interchange)

        if switch:
            assert config.switching_function
            assert config.switching_distance.m_as(unit.nanometer) == pytest.approx(0.8)
        else:
            assert not config.switching_function
            # No need to check the value of `switching_distance` ... right?


class TestSMIRNOFFDefaults(_BaseTest):
    @pytest.mark.parametrize("periodic", [True, False])
    def test_apply_smirnoff_defaults(self, sage, basic_top, periodic):
        from openff.units import unit

        from openff.interchange import Interchange

        interchange = Interchange.from_smirnoff(sage, basic_top)
        get_smirnoff_defaults(periodic=True).apply(interchange)

        for attr, value in zip(["cutoff", "switch_width"], [0.9, 0.1]):
            assert getattr(interchange["vdW"], attr).m_as(
                unit.nanometer,
            ) == pytest.approx(value)

        assert interchange["vdW"].method == "cutoff"

        if periodic:
            assert interchange["Electrostatics"].periodic_potential == _PME
        else:
            assert interchange["Electrostatics"].nonperiodic_potential == "Coulomb"


class TestIntermolDefaults(_BaseTest):
    @pytest.mark.parametrize("periodic", [True, False])
    def test_write_mdp(self, periodic):
        """
        https://github.com/shirtsgroup/InterMol/blob/master/intermol/tests/gromacs/grompp_vacuum.mdp
        https://github.com/shirtsgroup/InterMol/blob/master/intermol/tests/gromacs/grompp.mdp
        """

        get_intermol_defaults(
            periodic=periodic,
        ).write_mdp_file("tmp.mdp")

        options = parse_mdp("tmp.mdp")

        assert options["pbc"] == "xyz" if periodic else "no"

        assert options["coulombtype"] == "pme" if periodic else "cutoff"
        assert options["rcoulomb"] == "0.9" if periodic else "2.0"
        assert options["coulombmodifier"] == "none"

        assert options["vdwtype"] == "cutoff"
        assert options["rvdw"] == "0.9"
        assert options["vdwmodifier"] == "none"

        # This is set to "no" in the vacuum file, though it probably doesn't matter
        assert options["dispcorr"] == "ener"

        assert options["constraints"] == "none"

    def test_apply_intermol_defaults(self, sage, basic_top):
        from openff.units import unit

        from openff.interchange import Interchange

        interchange = Interchange.from_smirnoff(sage, basic_top)
        get_intermol_defaults(periodic=True).apply(interchange)

        assert interchange["vdW"].switch_width.m == 0.0

        for key in ["vdW", "Electrostatics"]:
            assert interchange[key].cutoff.m_as(unit.nanometer) == pytest.approx(0.9)
