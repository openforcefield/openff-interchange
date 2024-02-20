import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, unit
from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler
from openff.utilities import has_package, skip_if_missing

from openff.interchange._tests import MoleculeWithConformer, get_test_file_path

if has_package("openmm"):
    import openmm


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

        system = sage_with_trivalent_nitrogen.create_openmm_system(
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

        openff_system = openff_tip4p.create_openmm_system(water.to_topology())
        openff_virtual_site = openff_system.getVirtualSite(3)

        # Cannot directly compare weights because OpenMM (OutOfPlaneSite) and OpenFF (LocalCoordinatesSite)
        # use different implementations out of necessity, nor can we directly compare types
        assert isinstance(openff_virtual_site, openmm.LocalCoordinatesSite)

        # See table
        # https://docs.openforcefield.org/projects/recharge/en/latest/users/theory.html#generating-coordinates
        assert openff_virtual_site.getOriginWeights() == (1, 0, 0)
        assert openff_virtual_site.getXWeights() == (-1.0, 0.5, 0.5)
        assert openff_virtual_site.getYWeights() == (-1, 1, 0)

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

        openff_system = tip5p.create_openmm_system(water.to_topology())

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

            assert openff_virtual_site.getOriginWeights() == (1.0, 0, 0)
            assert openff_virtual_site.getXWeights() == (-1.0, 0.5, 0.5)
            assert openff_virtual_site.getYWeights() == (-1.0, 1.0, 0)

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


@skip_if_missing("openmm")
class TestOpenMMVirtualSiteExclusions:
    def test_tip5p_num_exceptions(self, water):
        tip5p = ForceField(get_test_file_path("tip5p.offxml"))

        out = tip5p.create_openmm_system(water.to_topology())

        # In a TIP5P water    expected exceptions include (total 10)
        #
        # V(3)  V(4)          Oxygen to hydrogens and particles (4)
        #    \ /                - (0, 1), (0, 2), (0, 3), (0, 4)
        #     O(0)            Hyrogens to virtual particles (4)
        #    / \                - (1, 3), (1, 4), (2, 3), (2, 4)
        # H(1)  H(2)          Hydrogens and virtual particles to each other (2)
        #                       - (1, 2), (3, 4)

        for force in out.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getNumExceptions() == 10

    def test_dichloroethane_exceptions(self, sage):
        """Test a case in which a parent's 1-4 exceptions must be 'imported'."""
        from openff.toolkit._tests.mocking import VirtualSiteMocking

        # This molecule has heavy atoms with indices (1-indexed) CL1, C2, C3, Cl4,
        # resulting in 1-4 interactions between the Cl-Cl pair and some Cl-H pairs
        dichloroethane = Molecule.from_mapped_smiles(
            "[Cl:1][C:2]([H:5])([H:6])[C:3]([H:7])([H:8])[Cl:4]",
        )

        # This parameter pulls 0.1 and 0.2e from Cl (parent) and C, respectively, and has
        # LJ parameters of 4 A, 3 kJ/mol
        parameter = VirtualSiteMocking.bond_charge_parameter("[Cl:1]-[C:2]")

        handler = VirtualSiteHandler(version="0.3")
        handler.add_parameter(parameter=parameter)

        sage.register_parameter_handler(handler)

        system = sage.create_openmm_system(dichloroethane.to_topology())

        assert system.isVirtualSite(8)
        assert system.isVirtualSite(9)

        non_bonded_force = [
            f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
        ][0]

        for exception_index in range(non_bonded_force.getNumExceptions()):
            p1, p2, q, sigma, epsilon = non_bonded_force.getExceptionParameters(
                exception_index,
            )
            if p2 == 8:
                # Parent Cl, adjacent C and its bonded H, and the 1-3 C
                if p1 in (0, 1, 2, 4, 5):
                    assert q._value == epsilon._value == 0.0
                # 1-4 Cl or 1-4 Hs
                if p1 in (3, 6, 7):
                    for value in (q, sigma, epsilon):
                        assert value._value != 0, (q, sigma, epsilon)
            if p2 == 9:
                if p1 in (3, 1, 2, 6, 7):
                    assert q._value == epsilon._value == 0.0
                if p1 in (0, 4, 5):
                    for value in (q, sigma, epsilon):
                        assert value._value != 0, (q, sigma, epsilon)

    def test_off_center_hydrogen_water(
        self,
        sage_with_off_center_hydrogen,
        water,
    ):
        """Reproduce oxygen case of issue #905."""
        import openmm

        system = sage_with_off_center_hydrogen.create_openmm_system(
            water.to_topology(),
        )

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                exceptions = [
                    force.getExceptionParameters(index)
                    for index in range(force.getNumExceptions())
                ]

        assert len(exceptions) == 10

        expected_pairs = {
            tuple(sorted(pair))
            for pair in (
                (0, 1),  # O-H
                (0, 2),  # O-H
                (0, 3),  # O-VS
                (0, 4),  # O-VS
                (1, 2),  # H-H
                (1, 3),  # H-VS (self)
                (2, 4),  # H-VS (self)
                (1, 4),  # H-VS (other)
                (2, 3),  # H-VS (other)
                (3, 4),  # VS-VS
            )
        }

        assert {tuple(sorted((p1, p2))) for p1, p2, _, _, _ in exceptions} == set(
            expected_pairs,
        )

        for _, _, charge, _, epsilon in exceptions:
            assert charge._value == 0.0
            assert epsilon._value == 0.0

    def test_off_center_hydrogen_methanol(
        self,
        sage_with_off_center_hydrogen,
    ):
        """Reproduce oxygen case of issue #905."""
        import openmm

        system = sage_with_off_center_hydrogen.create_openmm_system(
            MoleculeWithConformer.from_mapped_smiles(
                "[H:3][C:1]([H:4])([H:5])[O:2][H:6]",
            ).to_topology(),
        )

        # zero-indexed, though the order of virtual site indices is not guaranteed
        #
        #             H(3)*VS(7)
        #             |
        #  VS(6)*H(2) - C(0) - O(1) - H(5)*VS(9)
        #             |
        #             H(4)*VS(8)
        #
        #
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                exceptions = [
                    force.getExceptionParameters(index)
                    for index in range(force.getNumExceptions())
                ]

                break

        expected_zeroed_pairs = {
            tuple(sorted(pair))
            for pair in (
                (0, 1),  # C-O
                (0, 2),  # C-H
                (0, 3),  # C-H
                (0, 4),  # C-H
                (0, 5),  # C-H (through the oxygen)
                (0, 6),  # C-VS
                (0, 7),  # C-VS
                (0, 8),  # C-VS
                (0, 9),  # C-VS (through the oxygen)
                (1, 2),  # O-H (through the carbon)
                (1, 3),  # O-H (through the carbon)
                (1, 4),  # O-H (through the carbon)
                (1, 5),  # O-H
                (1, 6),  # O-VS (through the carbon)
                (1, 7),  # O-VS (through the carbon)
                (1, 8),  # O-VS (through the carbon)
                (1, 9),  # O-VS
                (2, 3),  # H-H
                (2, 4),  # H-H
                (2, 6),  # H-VS
                (2, 7),  # H-VS
                (2, 8),  # H-VS
                (3, 4),  # H-H
                (3, 6),  # H-VS
                (3, 7),  # H-VS
                (3, 8),  # H-VS
                (4, 6),  # H-VS
                (4, 7),  # H-VS
                (4, 8),  # H-VS
                (5, 9),  # H-VS
                (6, 7),  # VS-VS (within methyl)
                (6, 8),  # VS-VS (within methyl)
                (7, 8),  # VS-VS (within methyl)
            )
        }

        expected_14_pairs = {
            tuple(sorted(pair))
            for pair in (
                (2, 5),  # H-H (through the oxygen)
                (3, 5),  # H-H (through the oxygen)
                (4, 5),  # H-H (through the oxygen)
                (5, 6),  # H-VS (through the oxygen and carbon)
                (5, 7),  # H-VS (through the oxygen and carbon)
                (5, 8),  # H-VS (through the oxygen and carbon)
                (2, 9),  # H-VS (through the oxygen and carbon)
                (3, 9),  # H-VS (through the oxygen and carbon)
                (4, 9),  # H-VS (through the oxygen and carbon)
                (6, 9),  # VS-VS (through the oxygen and carbon)
                (7, 9),  # VS-VS (through the oxygen and carbon)
                (8, 9),  # VS-VS (through the oxygen and carbon)
            )
        }

        # the virtual sites on hydrogens don't carry charge, so they're only truly
        # zeroed if their vdW interactions are turned off
        assert {
            tuple(sorted((p1, p2)))
            for p1, p2, _, _, epsilon in exceptions
            if epsilon._value == 0.0
        } == set(
            expected_zeroed_pairs,
        )

        assert {
            tuple(sorted((p1, p2)))
            for p1, p2, _, _, epsilon in exceptions
            if epsilon._value != 0.0
        } == set(
            expected_14_pairs,
        )

        coul_14 = sage_with_off_center_hydrogen["Electrostatics"].scale14
        vdw_14 = sage_with_off_center_hydrogen["vdW"].scale14

        for p1, p2, charge_product, sigma, epsilon in exceptions:
            if tuple(sorted((p1, p2))) in expected_zeroed_pairs:
                assert charge_product._value == 0.0
                assert epsilon._value == 0.0
            elif tuple(sorted((p1, p2))) in expected_14_pairs:
                p1_parameters = force.getParticleParameters(p1)
                p2_parameters = force.getParticleParameters(p2)

                expected_charge = p1_parameters[0] * p2_parameters[0] * coul_14
                expected_sigma = (p1_parameters[1] + p2_parameters[1]) * 0.5

                expected_epsilon = (p1_parameters[2] * p2_parameters[2]) ** 0.5 * vdw_14

                assert charge_product._value == pytest.approx(expected_charge._value)

                assert sigma._value == pytest.approx(expected_sigma._value)

                assert epsilon._value == pytest.approx(expected_epsilon._value)

            else:
                raise Exception("Unexpected exception found")
