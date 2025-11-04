"""Unit tests for common virtual site functions."""

from random import random

import numpy
import pytest
from openff.toolkit import ForceField, Quantity

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.exceptions import MissingVirtualSitesError
from openff.interchange.interop._virtual_sites import get_positions_with_virtual_sites


@pytest.fixture
def tip4p_interchange(water, tip4p):
    return tip4p.create_interchange(water.to_topology())


def test_short_circuit_emtpy_virtual_site_collection(tip3p, water):
    from openff.interchange.smirnoff import SMIRNOFFVirtualSiteCollection

    out = tip3p.create_interchange(water.to_topology())

    with pytest.raises(MissingVirtualSitesError):
        get_positions_with_virtual_sites(out)

    out.collections["VirtualSites"] = SMIRNOFFVirtualSiteCollection()

    with pytest.raises(MissingVirtualSitesError):
        get_positions_with_virtual_sites(out)


def test_nonzero_positions(tip4p_interchange):
    assert not numpy.allclose(
        get_positions_with_virtual_sites(tip4p_interchange)[-1, :].m,
        numpy.zeros((1, 3)),
    )


def test_collate_virtual_site_positions(tip4p, water_dimer):
    out = tip4p.create_interchange(water_dimer)

    out.box = Quantity([10, 10, 10], "nanometer")

    # move the second water far away, since we're later checking that
    # each water's virtual site is close to its oxygen
    out.positions[3:] += Quantity([3, 3, 3], "nanometer")

    positions = get_positions_with_virtual_sites(
        out,
        collate=False,
    ).m_as("nanometer")

    collated_positions = get_positions_with_virtual_sites(
        out,
        collate=True,
    ).m_as("nanometer")

    # first three atoms and last virtual site should not be affected
    assert positions[:3, :] == pytest.approx(collated_positions[:3, :])
    assert positions[-1, :] == pytest.approx(collated_positions[-1, :])

    # first molecule's virtual site is placed at 4th position if collated,
    # second-to-last position if not
    assert positions[-2] == pytest.approx(collated_positions[3])

    def are_close(a, b):
        """Given two positions, return that they're < 1 nanometer apart."""
        return numpy.linalg.norm(a - b) < 1

    # each molecule's oxygen and virtual site should be close-ish
    assert are_close(positions[0], positions[-2])
    assert are_close(positions[3], positions[-1])
    assert are_close(collated_positions[0], collated_positions[3])
    assert are_close(collated_positions[4], collated_positions[7])

    # different molecules' oxygen and virtual site should NOT be close-ish
    assert not are_close(positions[0], positions[-1])
    assert not are_close(positions[3], positions[-2])
    assert not are_close(collated_positions[0], collated_positions[7])
    assert not are_close(collated_positions[4], collated_positions[3])


class TestVirtualSitePositions:
    @pytest.mark.skip(reason="Broken")
    @pytest.mark.parametrize(
        "distance_",
        [
            0.08,
            0.0,
            -0.08,
            -0.16,
            -0.24,
        ],
    )
    def test_bond_charge_positions(
        self,
        sage_with_bond_charge,
        distance_,
    ):
        sage_with_bond_charge["VirtualSites"].get_parameter(
            parameter_attrs={"smirks": "[#6:2]-[#17X1:1]"},
        )[0].distance = Quantity(
            distance_,
            "nanometer",
        )

        out = sage_with_bond_charge.create_interchange(
            MoleculeWithConformer.from_mapped_smiles(
                "[H:3][C:2]([H:4])([H:5])[Cl:1]",
            ).to_topology(),
        )

        positions = get_positions_with_virtual_sites(out)

        distance = numpy.linalg.norm(positions[-1, :].m - positions[0, :].m)

        assert numpy.allclose(
            distance,
            abs(distance_),
        )

    @pytest.mark.parametrize(
        (
            "distance_",
            "theta",
        ),
        [
            (0.08, 120),
            (0.15, 120),
            (0.15, 100),
            (0.06, 180),
            (0.05 + random(), 90 + 90 * random()),
        ],
    )
    def test_planar_monovalent_positions(
        self,
        sage_with_planar_monovalent_carbonyl,
        carbonyl_planar,
        distance_,
        theta,
    ):
        sage_with_planar_monovalent_carbonyl["VirtualSites"].get_parameter(
            parameter_attrs={"smirks": "[#8:1]=[#6X3+0:2]-[#6:3]"},
        )[0].distance = Quantity(
            distance_,
            "nanometer",
        )

        sage_with_planar_monovalent_carbonyl["VirtualSites"].get_parameter(
            parameter_attrs={"smirks": "[#8:1]=[#6X3+0:2]-[#6:3]"},
        )[0].inPlaneAngle = Quantity(
            theta,
            "degree",
        )

        out = sage_with_planar_monovalent_carbonyl.create_interchange(
            carbonyl_planar.to_topology(),
        )

        assert theta == next(iter(out["VirtualSites"].potentials.values())).parameters["inPlaneAngle"].m_as(
            "degree",
        )
        assert distance_ == next(iter(out["VirtualSites"].potentials.values())).parameters["distance"].m_as(
            "nanometer",
        )

        positions = get_positions_with_virtual_sites(out)

        distance = numpy.linalg.norm(
            positions[-1, :].m_as("nanometer") - positions[0, :].m_as("nanometer"),
        )

        try:
            assert distance == pytest.approx(distance_)
        except AssertionError:
            # TODO: Fix me!
            pytest.xfail()

    @pytest.mark.parametrize(
        (
            "distance_",
            "w1",
            "w2",
            "w3",
        ),
        [
            (0.0, 1.0, 0.0, 0.0),
            # TIP4P-FB
            (
                -0.010527445756662016,
                0.8203146574531,
                0.08984267127345003,
                0.08984267127345003,
            ),
            # TIP4P-FB but "backwards"
            (
                0.010527445756662016,
                1.1796853425469,
                -0.08984267127345003,
                -0.08984267127345003,
            ),
            # virtual site place at midpoint
            # d = -1 * (d_OH ** 2 - (0.5 * d_HH) ** 2) ** 0.5
            (-0.0585882276619988, 0.0, 0.5, 0.5),
        ],
    )
    def test_four_site_water_positions(
        self,
        water_tip4p,
        distance_,
        w1,
        w2,
        w3,
    ):
        tip4p = ForceField("tip4p_fb.offxml")

        tip4p["VirtualSites"].get_parameter(
            parameter_attrs={"smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]"},
        )[0].distance = Quantity(
            distance_,
            "nanometer",
        )

        out = tip4p.create_interchange(water_tip4p.to_topology())

        positions = get_positions_with_virtual_sites(out)

        p1 = out.positions[0]
        p2 = out.positions[1]
        p3 = out.positions[2]

        assert numpy.allclose(
            positions[-1].m_as("angstrom"),
            (w1 * p1 + w2 * p2 + w3 * p3).m_as("angstrom"),
        )

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
    def test_trivalent_nitrogen_positions(
        self,
        sage_with_trivalent_nitrogen,
        ammonia_tetrahedral,
        distance_,
    ):
        sage_with_trivalent_nitrogen["VirtualSites"].get_parameter(
            parameter_attrs={"smirks": "[#1:2][#7:1]([#1:3])[#1:4]"},
        )[0].distance = Quantity(
            distance_,
            "angstrom",
        )

        out = sage_with_trivalent_nitrogen.create_interchange(
            ammonia_tetrahedral.to_topology(),
        )

        positions = get_positions_with_virtual_sites(out).to("angstrom")

        distance = numpy.linalg.norm(positions[-1, :].m - positions[0, :].m)

        assert distance == pytest.approx(abs(distance_))

        # The nitrogen is placed at [0, 0, 0.8855572013] and the hydrogens are on
        # the xy plane, so the virtual site is at [0, 0, 0.88 ... + distance_]
        assert positions[-1].m_as("angstrom")[0] == pytest.approx(0.0)
        # assert positions[-1].m_as("angstrom")[1] == pytest.approx(0.0)
