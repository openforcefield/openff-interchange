"""Unit tests for common virtual site functions."""
import numpy
import pytest
from openff.toolkit import ForceField
from openff.units import Quantity, unit

from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.exceptions import MissingVirtualSitesError
from openff.interchange.interop._virtual_sites import get_positions_with_virtual_sites


@pytest.fixture()
def tip4p_interchange(water, tip4p):
    return tip4p.create_interchange(water.to_topology())


def test_short_circuit_emtpy_virtual_site_collection(tip3p, water):
    from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection

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


class TestVirtualSitePositions:
    @pytest.mark.parametrize(
        (
            "distance_",
            "w1",
            "w2",
        ),
        [
            (0.8, 1.5, -0.5),
            (0.0, 1.0, 0.0),
            (-0.8, 0.5, 0.5),
            (-1.6, 0.0, 1.0),
            (-2.4, -0.5, 1.5),
        ],
    )
    def test_bond_charge_positions(
        self,
        sage_with_bond_charge,
        distance_,
        w1,
        w2,
    ):
        sage_with_bond_charge["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        out = sage_with_bond_charge.create_interchange(
            MoleculeWithConformer.from_mapped_smiles(
                "[H:3][C:2]([H:4])([H:5])[Cl:1]",
            ).to_topology(),
        )

        positions = get_positions_with_virtual_sites(out)

        p1 = out.positions[0]
        p2 = out.positions[1]

        assert numpy.allclose(
            positions[-1].m_as(unit.angstrom),
            (w1 * p1 + w2 * p2).m_as(unit.angstrom),
        )

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
    def test_four_site_water_positions(
        self,
        water_tip4p,
        distance_,
        w1,
        w2,
        w3,
    ):
        tip4p = ForceField("tip4p_fb.offxml")

        tip4p["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        out = tip4p.create_interchange(water_tip4p.to_topology())

        positions = get_positions_with_virtual_sites(out)

        p1 = out.positions[0]
        p2 = out.positions[1]
        p3 = out.positions[2]

        assert numpy.allclose(
            positions[-1].m_as(unit.angstrom),
            (w1 * p1 + w2 * p2 + w3 * p3).m_as(unit.angstrom),
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
        sage_with_trivalent_nitrogen["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        out = sage_with_trivalent_nitrogen.create_interchange(
            ammonia_tetrahedral.to_topology(),
        )

        original = ammonia_tetrahedral.conformers[0]
        positions = get_positions_with_virtual_sites(out).to(unit.angstrom)

        # The nitrogen is placed at [0, 0, 0.8855572013] and the hydrogens are on
        # the xy plane, so the virtual site is at [0, 0, 0.88 ... + distance_]
        assert positions[-1].m_as(unit.angstrom)[0] == 0.0
        assert positions[-1].m_as(unit.angstrom)[1] == 0.0
        assert positions[-1].m_as(unit.angstrom)[2] == 0.8855572013 + distance_

        d_n_vs = positions[-1] - original[0]
        d_n_h = [
            positions[1] - original[0],
            positions[2] - original[0],
            positions[3] - original[0],
        ]

        assert numpy.linalg.norm(
            d_n_vs.m_as(unit.angstrom),
        ) == pytest.approx(numpy.abs(distance_))

        for d in d_n_h[1:]:
            assert numpy.linalg.norm(
                d.m_as(unit.angstrom),
            ) == pytest.approx(numpy.linalg.norm(d_n_h[0].m_as(unit.angstrom)))
