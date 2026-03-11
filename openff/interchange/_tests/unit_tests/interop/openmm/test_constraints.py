import pytest
from openff.toolkit import Quantity
from openff.toolkit.topology import ValenceDict
from openff.toolkit.typing.engines.smirnoff.parameters import ConstraintHandler
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError

from openff.interchange.exceptions import MissingParametersError


@pytest.fixture
def distanceless_bond_constraints():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*:2]",  # all bonds!
            "id": "distanceless_bond",
        },
    )

    return handler


@pytest.fixture
def bond_constraints_with_distance():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*:2]",
            "id": "bond_with_distance",
            "distance": Quantity("0.123456789 nm"),
        },
    )

    return handler


@pytest.fixture
def distanceless_angle_constraints():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*]~[*:2]",  # all angles!
            "id": "distanceless_angle",
        },
    )

    return handler


@pytest.fixture
def angle_constraints_with_distance():
    handler = ConstraintHandler(version=0.3)
    handler.add_parameter(
        {
            "smirks": "[*:1]~[*]~[*:2]",  # all angles!
            "id": "angle_with_distance",
            "distance": Quantity("0.22222222 nm"),
        },
    )

    return handler


class TestConstraints:
    def test_with_bonds_with_distanceless_constraints(
        self,
        fresh_sage,
        distanceless_bond_constraints,
        ethanol,
    ):
        """Bonds specified, constraints without distance, length taken from bonds."""
        openmm = pytest.importorskip("openmm")

        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(distanceless_bond_constraints)

        # add in bond parameters which are also constrained, since it's easier and a little safer
        # than looking up what they *should* be
        system = fresh_sage.create_interchange(ethanol.to_topology()).to_openmm_system(add_constrained_forces=True)

        assert system.getNumConstraints() == ethanol.n_bonds

        constraint_distances = ValenceDict()
        bond_distances = ValenceDict()

        for constraint_index in range(system.getNumConstraints()):
            atom1_index, atom2_index, distance = system.getConstraintParameters(constraint_index)

            assert "0.123456789" not in str(distance)

            constraint_distances[(atom1_index, atom2_index)] = distance

        bond_force = next(force for force in system.getForces() if isinstance(force, openmm.HarmonicBondForce))

        for bond_index in range(bond_force.getNumBonds()):
            atom1_index, atom2_index, length, _ = bond_force.getBondParameters(bond_index)

            bond_distances[(atom1_index, atom2_index)] = length

        # some bonds (C-C, C-O) don't have constraints, so can't compare entire dicts
        # but all constraints should have corresponding bond parameters
        for constrained_pair, constrained_distance in constraint_distances.items():
            assert constrained_distance == bond_distances[constrained_pair]

    def test_constraint_distances_override_bond_distances(
        self,
        fresh_sage,
        bond_constraints_with_distance,
        ethanol,
    ):
        """Bonds specified, constraints without distance, length taken from bonds."""
        openmm = pytest.importorskip("openmm")

        # replace original constraints with a single wildcard constraint, since it's
        # easier and a little safer than looking up what each distance  *should* be
        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(bond_constraints_with_distance)
        system = fresh_sage.create_interchange(ethanol.to_topology()).to_openmm_system(add_constrained_forces=True)

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
        fresh_sage,
        distanceless_bond_constraints,
        ethanol,
    ):
        """When constraints are speicifed without distances, but no bonds are present, error."""
        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(distanceless_bond_constraints)
        fresh_sage.deregister_parameter_handler("Bonds")

        # this test short-circuits before anything OpenMM is called, so it could live elsewhere
        with pytest.raises(
            MissingParametersError,
            match=r"The distance of this constraint is not specified.",
        ):
            fresh_sage.create_openmm_system(ethanol.to_topology())

    def test_constraints_with_distances_without_bonds(
        self,
        fresh_sage,
        bond_constraints_with_distance,
        ethanol,
    ):
        """When constraints are speicifed with distances, but no bonds are present, still sets constraints."""
        openmm = pytest.importorskip("openmm")

        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(bond_constraints_with_distance)
        fresh_sage.deregister_parameter_handler("Bonds")

        system = fresh_sage.create_interchange(ethanol.to_topology()).to_openmm_system()

        assert system.getNumConstraints() == ethanol.n_bonds

        for constraint_index in range(system.getNumConstraints()):
            _, _, distance = system.getConstraintParameters(constraint_index)

            assert "0.12345678" in str(distance)

        for force in system.getForces():
            if type(force) is openmm.HarmonicBondForce:
                raise Exception("there should not be a bond force")

    @pytest.mark.parametrize("remove_angles", [True, False])
    def test_angles_with_distanceless_constraints(
        self,
        distanceless_angle_constraints,
        fresh_sage,
        ethanol,
        remove_angles,
    ):
        """
        Test that angle-like constraints without specified distance raise an error,
        whether or not there are angle parameters.
        """
        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(distanceless_angle_constraints)

        if remove_angles:
            fresh_sage.deregister_parameter_handler("Angles")

        with pytest.raises(
            SMIRNOFFSpecError,
            match=r"0, 2.*unsupported in the SMIRNOFF specification",
        ):
            fresh_sage.create_openmm_system(ethanol.to_topology())

    def test_constraint_distances_override_angle_geometry(
        self,
        fresh_sage,
        ethanol,
        angle_constraints_with_distance,
    ):
        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(angle_constraints_with_distance)

        system = fresh_sage.create_interchange(ethanol.to_topology()).to_openmm_system(add_constrained_forces=True)

        assert system.getNumConstraints() == ethanol.n_angles

        for constraint_index in range(system.getNumConstraints()):
            _, _, distance = system.getConstraintParameters(constraint_index)

            assert "0.222222" in str(distance)

        # angle parameters don't store 1-3 distance, so can't cleanly compare geometries vs. constraint distance

    def test_constraints_with_distances_without_angles(
        self,
        fresh_sage,
        ethanol,
        angle_constraints_with_distance,
    ):
        """When constraints are speicifed with distances, but no angles are present, still sets constraints."""
        openmm = pytest.importorskip("openmm")

        fresh_sage.deregister_parameter_handler("Constraints")
        fresh_sage.register_parameter_handler(angle_constraints_with_distance)
        fresh_sage.deregister_parameter_handler("Angles")

        system = fresh_sage.create_interchange(ethanol.to_topology()).to_openmm_system()

        assert system.getNumConstraints() == ethanol.n_angles

        for constraint_index in range(system.getNumConstraints()):
            _, _, distance = system.getConstraintParameters(constraint_index)

            assert "0.222222" in str(distance)

        for force in system.getForces():
            if type(force) is openmm.HarmonicAngleForce:
                raise Exception("there should not be an angle force")
