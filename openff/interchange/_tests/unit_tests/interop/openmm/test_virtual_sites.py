import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import Quantity, unit
from openff.utilities.testing import skip_if_missing


@skip_if_missing("openmm")
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
    def test_bond_charge_weights(
        self,
        sage_with_bond_charge,
        water,
        distance_,
        expected_w1,
        expected_w2,
    ):
        import openmm

        sage_with_bond_charge["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        system = sage_with_bond_charge.create_openmm_system(
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


@skip_if_missing("openmm")
class TestFourSiteWater:
    @pytest.mark.parametrize(
        (
            "distance_",
            "expected_w1",
            "expected_w2",
            "expected_w3",
        ),
        [
            (0.0, 1.0, 0.0, 0.0),
            # TIP4P-FB
            (
                -0.10527445756662016,
                0.8203146574531,
                0.08984267127345003,
                0.08984267127345003,
            ),
            # TIP4P-FB but "backwards"
            (
                0.10527445756662016,
                1.1796853425469,
                -0.08984267127345003,
                -0.08984267127345003,
            ),
            # virtual site place at midpoint
            # d = -1 * (d_OH ** 2 - (0.5 * d_HH) ** 2) ** 0.5
            (-0.585882276619988, 0.0, 0.5, 0.5),
        ],
    )
    def test_weights(
        self,
        water,
        distance_,
        expected_w1,
        expected_w2,
        expected_w3,
    ):
        import openmm

        tip4p = ForceField("tip4p_fb.offxml")

        tip4p["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        system = tip4p.create_openmm_system(water.to_topology())

        assert system.getNumParticles() == 4

        for index in range(3):
            assert system.getParticleMass(index)._value > 0.0

            try:
                system.getVirtualSite(index)
            except openmm.OpenMMException as error:
                # This method raises an error, not sure if there's another method
                # that just returns a boolean as to whether or not it's a virtual site
                assert "This particle is not a virtual site" in str(error)

        assert system.getParticleMass(3)._value == 0.0

        virtual_site = system.getVirtualSite(3)

        assert isinstance(virtual_site, openmm.ThreeParticleAverageSite)

        assert virtual_site.getParticle(0) == 0
        assert virtual_site.getParticle(1) == 1
        assert virtual_site.getParticle(2) == 2

        assert virtual_site.getWeight(0) == pytest.approx(expected_w1)
        assert virtual_site.getWeight(1) == pytest.approx(expected_w2)
        assert virtual_site.getWeight(2) == pytest.approx(expected_w3)


@skip_if_missing("openmm")
class TestTIP4PVsOpenMM:
    def test_compare_tip4pfb_openmm(self, water):
        import openmm.app
        from openff.toolkit import ForceField

        openmm_tip4pfb = openmm.app.ForceField("tip4pfb.xml")
        openmm_topology = water.to_topology().to_openmm()
        openmm_positions = water.conformers[0].to_openmm()

        modeller = openmm.app.Modeller(
            openmm_topology,
            openmm_positions,
        )

        modeller.addExtraParticles(openmm_tip4pfb)

        openmm_virtual_site = openmm_tip4pfb.createSystem(
            modeller.topology,
        ).getVirtualSite(3)
        openmm_weights = [openmm_virtual_site.getWeight(index) for index in range(3)]

        openff_tip4p = ForceField("tip4p_fb.offxml")

        openff_virtual_site = openff_tip4p.create_openmm_system(
            water.to_topology(),
        ).getVirtualSite(3)
        openff_weights = [openff_virtual_site.getWeight(index) for index in range(3)]

        assert type(openmm_virtual_site) is type(openff_virtual_site)

        for w_openmm, w_openff in zip(openmm_weights, openff_weights):
            assert w_openmm == pytest.approx(w_openff)
