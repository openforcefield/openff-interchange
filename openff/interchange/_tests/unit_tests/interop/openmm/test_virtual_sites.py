import numpy
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import Quantity, unit
from openff.utilities.testing import skip_if_missing


@skip_if_missing("openmm")
class TestBondChargeVirtualSite:
    @pytest.mark.parametrize(
        "distance_",
        [
            0.8,
            0.0,
            -0.8,
            -1.6,
            -2.4,
        ],
    )
    def test_bond_charge_weights(
        self,
        sage_with_bond_charge,
        distance_,
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

        assert virtual_site.getParticle(0) == 0
        assert virtual_site.getParticle(1) == 1

        numpy.testing.assert_allclose(
            Quantity(
                [
                    -1 * distance_,
                    0,
                    0,
                ],
                unit.angstrom,
            ).m_as(unit.angstrom),
            numpy.asarray(
                virtual_site.getLocalPosition().value_in_unit(openmm.unit.angstrom),
            ),
        )


@skip_if_missing("openmm")
class TestTrivalentLonePairVirtualSite:
    def test_basic(
        self,
        sage_with_trivalent_nitrogen,
        ammonia_tetrahedral,
    ):
        import openmm

        system: openmm.System = sage_with_trivalent_nitrogen.create_openmm_system(
            ammonia_tetrahedral.to_topology(),
        )

        assert isinstance(system.getVirtualSite(4), openmm.LocalCoordinatesSite)


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

        assert isinstance(virtual_site, openmm.LocalCoordinatesSite)

        assert virtual_site.getParticle(0) == 0
        assert virtual_site.getParticle(1) == 1
        assert virtual_site.getParticle(2) == 2

        # Negative because SMIRNOFF points the virtual site "away" from hydrogens but the
        # coordinate system is pointed "towards" them; 0.1 because Angstrom -> nanometer
        assert distance_ * -0.1 == pytest.approx(
            virtual_site.getLocalPosition()[0].value_in_unit(openmm.unit.nanometer),
        )

        assert pytest.approx(virtual_site.getLocalPosition()[1]._value) == 0.0
        assert pytest.approx(virtual_site.getLocalPosition()[2]._value) == 0.0


@skip_if_missing("openmm")
class TestTIP4PVsOpenMM:
    def test_compare_tip4pfb_openmm(self, water):
        import openmm.app
        from openff.toolkit import ForceField

        if False:
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
            openmm_weights = [  # noqa
                openmm_virtual_site.getWeight(index) for index in range(3)
            ]

        openff_tip4p = ForceField("tip4p_fb.offxml")

        openff_system = openff_tip4p.create_openmm_system(
            water.to_topology(),
        )
        openff_virtual_site = openff_system.getVirtualSite(3)

        # Cannot directly compare weights because OpenMM (OutOfPlaneSite) and OpenFF (LocalCoordinatesSite)
        # use different implementations out of necessity, nor can we directly compare types
        assert isinstance(openff_virtual_site, openmm.LocalCoordinatesSite)

        # See table
        # https://docs.openforcefield.org/projects/recharge/en/latest/users/theory.html#generating-coordinates
        assert openff_virtual_site.getOriginWeights() == (0, 1, 0)
        assert openff_virtual_site.getXWeights() == (0.5, -1.0, 0.5)
        assert openff_virtual_site.getYWeights() == (1, -1, 0)

        assert openff_virtual_site.getParticle(0) == 0
        assert openff_virtual_site.getParticle(1) in (1, 2)
        assert openff_virtual_site.getParticle(2) in (2, 1)

        # This local position should directly match the distance (O-VS) one would draw by hand (0.10527... Angstrom)
        # https://github.com/pandegroup/tip3p-tip4p-fb/blob/master/AMBER/dat/leap/parm/frcmod.tip4pfb#L7
        # https://github.com/openforcefield/openff-forcefields/blob/2023.08.0/openforcefields/offxml/tip4p_fb-1.0.0.offxml#L137
        assert openff_virtual_site.getLocalPosition()[0].value_in_unit(
            openmm.unit.nanometer,
        ) == pytest.approx(0.010527445756662016)
        assert openff_virtual_site.getLocalPosition()[1]._value == 0.0
        assert openff_virtual_site.getLocalPosition()[2]._value == 0.0

        # TODO: Also doubly compare geometry to OpenMM result


@skip_if_missing("openmm")
class TestTIP5PVsOpenMM:
    def test_compare_tip5p_openmm(self, water, tip5p):
        import openmm.app

        if False:
            openmm_tip5p = openmm.app.ForceField("tip5p.xml")
            openmm_topology = water.to_topology().to_openmm()
            openmm_positions = water.conformers[0].to_openmm()

            modeller = openmm.app.Modeller(
                openmm_topology,
                openmm_positions,
            )

            modeller.addExtraParticles(openmm_tip5p)

            openmm_system = openmm_tip5p.createSystem(
                modeller.topology,
                removeCMMotion=False,
            )

            openmm_virtual_sites = [  # noqa
                openmm_system.getVirtualSite(3),
                openmm_system.getVirtualSite(4),
            ]

        openff_system = tip5p.create_openmm_system(
            water.to_topology(),
        )

        openff_virtual_sites = [
            openff_system.getVirtualSite(3),
            openff_system.getVirtualSite(4),
        ]

        # Cannot directly compare weights because OpenMM (OutOfPlaneSite) and OpenFF (LocalCoordinatesSite)
        # use different implementations out of necessity

        # See table
        # https://docs.openforcefield.org/projects/recharge/en/latest/users/theory.html#generating-coordinates
        for openff_virtual_site in openff_virtual_sites:
            assert isinstance(openff_virtual_site, openmm.LocalCoordinatesSite)

            assert openff_virtual_site.getOriginWeights() == (0, 1, 0)
            assert openff_virtual_site.getXWeights() == (0.5, -1.0, 0.5)
            assert openff_virtual_site.getYWeights() == (1, -1, 0)

            assert openff_virtual_site.getLocalPosition()[0].value_in_unit(
                openmm.unit.nanometer,
            ) == pytest.approx(-0.040415127656087124)
            assert openff_virtual_site.getLocalPosition()[1]._value == 0.0
            assert openff_virtual_site.getLocalPosition()[2].value_in_unit(
                openmm.unit.nanometer,
            ) == pytest.approx(0.0571543301644082)

        # Both particles should be indexed as O-H-H, but the order of the hydrogens should be flipped
        # (this is how the local positions are the same but "real" space positions are flipped)
        assert openff_virtual_sites[0].getParticle(0) == openff_virtual_sites[
            1
        ].getParticle(0)
        assert openff_virtual_sites[0].getParticle(1) == openff_virtual_sites[
            1
        ].getParticle(2)
        assert openff_virtual_sites[0].getParticle(2) == openff_virtual_sites[
            1
        ].getParticle(1)

        # TODO: Also doubly compare geometry to OpenMM result


# TODO: Port xml_ff_virtual_sites_monovalent from toolkit
