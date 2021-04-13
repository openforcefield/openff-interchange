import numpy as np
import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from simtk import unit as simtk_unit

from openff.system.exceptions import MissingDependencyError
from openff.system.tests.base_test import BaseTest
from openff.system.utils import (
    compare_forcefields,
    get_partial_charges_from_openmm_system,
    pint_to_simtk,
    requires_package,
    simtk_to_pint,
    unwrap_list_of_pint_quantities,
)

simtk_quantitites = [
    4.0 * simtk_unit.nanometer,
    5.0 * simtk_unit.angstrom,
    1.0 * simtk_unit.elementary_charge,
]

pint_quantities = [
    4.0 * unit.nanometer,
    5.0 * unit.angstrom,
    1.0 * unit.elementary_charge,
]


@pytest.mark.parametrize(
    "simtk_quantity,pint_quantity",
    [(s, p) for s, p in zip(simtk_quantitites, pint_quantities)],
)
def test_simtk_to_pint(simtk_quantity, pint_quantity):
    """Test conversion from SimTK Quantity to pint Quantity."""
    converted_pint_quantity = simtk_to_pint(simtk_quantity)

    assert pint_quantity == converted_pint_quantity


def test_simtk_list_of_quantities_to_pint():
    """Test conversion from Quantity lists, lists of Quantity"""
    list_of_quantities = [val * simtk_unit.meter for val in range(10)]
    quantity_list = simtk_unit.meter * [val for val in range(10)]

    assert list_of_quantities != quantity_list
    assert all(simtk_to_pint(list_of_quantities) == simtk_to_pint(quantity_list))


def test_pint_to_simtk():
    """Test conversion from pint Quantity to SimTK Quantity."""
    q = 5.0 / unit.nanometer
    assert pint_to_simtk(q) == 0.5 / simtk_unit.angstrom


class TestUtils(BaseTest):
    def test_compare_forcefields(self, parsley):
        parsley_name = "openff-1.0.0.offxml"
        compare_forcefields(parsley, parsley)
        compare_forcefields(ForceField(parsley_name), parsley)
        compare_forcefields(parsley, ForceField(parsley_name))
        compare_forcefields(ForceField(parsley_name), ForceField(parsley_name))

    def test_unwrap_quantities(self):
        wrapped = [1 * unit.m, 1.5 * unit.m]
        unwrapped = [1, 1.5] * unit.m

        assert all(unwrapped == unwrap_list_of_pint_quantities(wrapped))


class TestOpenMM(BaseTest):
    def test_openmm_partial_charges(self, argon_ff, argon_top):
        omm_system = argon_ff.create_openmm_system(argon_top)
        partial_charges = get_partial_charges_from_openmm_system(omm_system)

        # assert isinstance(partial_charges, unit.Quantity)
        # assert partial_charges.units == unit.elementary_charge
        assert isinstance(partial_charges, list)
        assert np.allclose(partial_charges, np.zeros(4))  # .magnitude


def test_requires_package():
    """Test the @requires_package decorator"""

    @requires_package("re")
    def fn_installed():
        pass

    fn_installed()

    @requires_package("foobar")
    def fn_missing():
        pass

    with pytest.raises(MissingDependencyError, match="foobar"):
        fn_missing()
