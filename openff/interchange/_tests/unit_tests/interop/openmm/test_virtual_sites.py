import openmm
import pytest
from openff.toolkit.typing.engines.smirnoff.parameters import BondType, VirtualSiteType
from openff.units import Quantity, unit

from openff.interchange._tests import Molecule

# Test that


class TestBondChargeVirtualSite:
    @pytest.mark.parametrize(
        (
            "distance_",
            "expected_w1",
            "expected_w2",
        ),
        [
            (0.8, 1.5, -0.5),
            (0.0, 1.0, 0.0),
            (-0.8, 0.5, 0.5),
            (-1.6, 0.0, 1.0),
            (-2.4, -0.5, 1.5),
        ],
    )
    def test_bond_charge_geometry(
        self,
        sage,
        distance_,
        expected_w1,
        expected_w2,
    ):
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
                distance=Quantity(distance_, unit.angstrom),
                charge_increment1="0.0 * elementary_charge ** 1",
                charge_increment2="0.0 * elementary_charge ** 1",
            ),
        )

        system = sage.create_openmm_system(
            Molecule.from_mapped_smiles(
                "[H:3][C:2]([H:4])([H:5])[Cl:1]",
            ).to_topology(),
        )

        assert system.getNumParticles() == 6

        for index in range(5):
            assert system.getParticleMass(index)._value > 0.0

            try:
                system.getVirtualSite(index)
            except openmm.OpenMMException as error:
                # This method raises an error, not sure if there's another method
                # that just returns a boolean as to whether or not it's a virtual site
                assert "This particle is not a virtual site" in str(error)

        assert system.getParticleMass(5)._value == 0.0

        virtual_site = system.getVirtualSite(5)

        assert isinstance(virtual_site, openmm.TwoParticleAverageSite)

        assert virtual_site.getParticle(0) == 0
        assert virtual_site.getParticle(1) == 1

        assert virtual_site.getWeight(0) == pytest.approx(expected_w1)
        assert virtual_site.getWeight(1) == pytest.approx(expected_w2)
