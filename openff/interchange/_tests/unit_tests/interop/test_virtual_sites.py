"""Unit tests for common virtual site functions."""
import numpy
import pytest

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
