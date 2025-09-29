import pytest
from openff.toolkit import Quantity
from openff.toolkit.topology import ValenceDict
from openff.toolkit.typing.engines.smirnoff.parameters import ConstraintHandler

from openff.interchange.exceptions import MissingParametersError


@pytest.fixture
def distanceless_constraints():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*:2]",  # all bonds!
            "id": "distanceless",
        },
    )

    return handler


@pytest.fixture
def constraints_with_distance():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*:2]",
            "id": "distanceless",
            "distance": Quantity("0.123456789 nm"),
        },
    )

    return handler


@pytest.fixture
def force_field_with_distanceless_constraints(sage):
    """Force field with very different parameters for some particular bonds, angles, and constraints."""
    sage.dereg


class TestConstraints:
    def test_with_bonds_with_distanceless_constraints(
        self,
        sage,
        distanceless_constraints,
        ethanol,
    ):
        """Bonds specified, constraints without distance, length taken from bonds."""
        openmm = pytest.importorskip("openmm")

        sage.deregister_parameter_handler("Constraints")
        sage.register_parameter_handler(distanceless_constraints)

        # add in bond parameters which are also constrained, since it's easier and a little safer
        # than looking up what they *should* be
        system = sage.create_interchange(ethanol.to_topology()).to_openmm_system(add_constrained_forces=True)

        assert system.getNumConstraints() == ethanol.n_bonds

        constraint_distances = ValenceDict()
        bond_distances = ValenceDict()

        for constraint_index in range(system.getNumConstraints()):
            atom1_index, atom2_index, distance = system.getConstraintParameters(constraint_index)

            assert "0.123456789" not in str(distance)

            constraint_distances[(atom1_index, atom2_index)] = distance

        for force in system.getForces():
            if type(force) is not openmm.HarmonicBondForce:
                continue
            for bond_index in range(force.getNumBonds()):
                atom1_index, atom2_index, length, _ = force.getBondParameters(bond_index)

                bond_distances[(atom1_index, atom2_index)] = length

        # some bonds (C-C, C-O) don't have constraints, so can't compare entire dicts
        # but all constraints should have corresponding bond parameters
        for constrained_pair, constrained_distance in constraint_distances.items():
            assert constrained_distance == bond_distances[constrained_pair]

    def test_constraint_distances_override_bond_distances(
        self,
        sage,
        constraints_with_distance,
        ethanol,
    ):
        """Bonds specified, constraints without distance, length taken from bonds."""
        openmm = pytest.importorskip("openmm")

        sage.deregister_parameter_handler("Constraints")
        sage.register_parameter_handler(constraints_with_distance)

        # add in bond parameters which are also constrained, since it's easier and a little safer
        # than looking up what they *should* be
        system = sage.create_interchange(ethanol.to_topology()).to_openmm_system(add_constrained_forces=True)

        assert system.getNumConstraints() == ethanol.n_bonds

        for constraint_index in range(system.getNumConstraints()):
            _, _, distance = system.getConstraintParameters(constraint_index)

            assert "0.12345678" in str(distance)

        for force in system.getForces():
            if type(force) is not openmm.HarmonicBondForce:
                continue
            for bond_index in range(force.getNumBonds()):
                _, _, length, _ = force.getBondParameters(bond_index)

                assert "0.12345678" not in str(length)

    def test_distanceless_constraints_without_bonds_error(
        self,
        sage,
        distanceless_constraints,
        ethanol,
    ):
        """When constraints are speicifed without distances, but no bonds are present, error."""
        sage.deregister_parameter_handler("Constraints")
        sage.register_parameter_handler(distanceless_constraints)
        sage.deregister_parameter_handler("Bonds")

        # this test short-circuits before anything OpenMM is called, so it could live elsewhere
        with pytest.raises(
            MissingParametersError,
            match=r"The distance of this constraint is not specified.",
        ):
            sage.create_openmm_system(ethanol.to_topology())

    def test_constraints_with_distances_without_bonds(
        self,
        sage,
        constraints_with_distance,
        ethanol,
    ):
        """When constraints are speicifed with distances, but no bonds are present, still sets constraints."""
        openmm = pytest.importorskip("openmm")

        sage.deregister_parameter_handler("Constraints")
        sage.register_parameter_handler(constraints_with_distance)
        sage.deregister_parameter_handler("Bonds")

        system = sage.create_interchange(ethanol.to_topology()).to_openmm_system()

        assert system.getNumConstraints() == ethanol.n_bonds

        for constraint_index in range(system.getNumConstraints()):
            _, _, distance = system.getConstraintParameters(constraint_index)

            assert "0.12345678" in str(distance)

        for force in system.getForces():
            if type(force) is openmm.HarmonicBondForce:
                raise Exception("there should not be a bond force")
