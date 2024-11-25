from collections import defaultdict
from typing import TYPE_CHECKING

from openff.utilities.utilities import has_package, requires_package

if TYPE_CHECKING or has_package("openmm"):
    import openmm
    import openmm.unit

    kj_nm2_mol = openmm.unit.kilojoule_per_mole / openmm.unit.nanometer**2
    kj_rad2_mol = openmm.unit.kilojoule_per_mole / openmm.unit.radian**2


@requires_package("openmm")
def get_14_scaling_factors(omm_sys: "openmm.System") -> tuple[list, list]:
    """Find the 1-4 scaling factors as they are applied to an OpenMM System."""
    nonbond_force = [f for f in omm_sys.getForces() if type(f) is openmm.NonbondedForce][0]

    vdw_14 = list()
    coul_14 = list()

    for exception_idx in range(nonbond_force.getNumExceptions()):
        i, j, q, _, eps = nonbond_force.getExceptionParameters(exception_idx)

        # Trust that q == 0 covers the cases of 1-2, 1-3, and truly being 0
        if q._value != 0:
            q_i = nonbond_force.getParticleParameters(i)[0]
            q_j = nonbond_force.getParticleParameters(j)[0]
            coul_14.append(q / (q_i * q_j))

        if eps._value != 0:
            eps_i = nonbond_force.getParticleParameters(i)[2]
            eps_j = nonbond_force.getParticleParameters(j)[2]
            vdw_14.append(eps / (eps_i * eps_j) ** 0.5)

    return coul_14, vdw_14


def _create_torsion_dict(torsion_force) -> dict[tuple[int], list[tuple]]:
    torsions = defaultdict(list)

    for i in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(i)
        key = (p1, p2, p3, p4)
        torsions[key]
        torsions[key].append((periodicity, phase, k))

    return torsions


def _create_bond_dict(bond_force):
    bonds = dict()

    for i in range(bond_force.getNumBonds()):
        p1, p2, length, k = bond_force.getBondParameters(i)
        key = (p1, p2)
        bonds[key] = (length, k)

    return bonds


def _create_angle_dict(angle_force):
    angles = dict()

    for i in range(angle_force.getNumAngles()):
        p1, p2, p3, theta, k = angle_force.getAngleParameters(i)
        key = (p1, p2, p3)
        angles[key] = (theta, k)

    return angles


@requires_package("openmm")
def _compare_individual_torsions(x, y):
    assert x[0] == y[0]
    assert x[1] == y[1]
    assert (x[2] - y[2]) < 1e-15 * openmm.unit.kilojoule_per_mole


def _compare_torsion_forces(force1, force2):
    sorted1 = _create_torsion_dict(torsion_force=force1)
    sorted2 = _create_torsion_dict(torsion_force=force2)

    assert sum(len(v) for v in sorted1.values()) == force1.getNumTorsions()
    assert sum(len(v) for v in sorted2.values()) == force2.getNumTorsions()
    assert len(sorted1) == len(sorted2)

    for key in sorted1:
        for i in range(len(sorted1[key])):
            _compare_individual_torsions(sorted1[key][i], sorted2[key][i])


@requires_package("openmm")
def _compare_bond_forces(force1, force2):
    assert force1.getNumBonds() == force2.getNumBonds()

    bonds1 = _create_bond_dict(force1)
    bonds2 = _create_bond_dict(force2)

    for key in bonds1:
        length_diff = bonds2[key][0] - bonds1[key][0]
        assert abs(length_diff) < 1e-15 * openmm.unit.nanometer, f"Bond lengths differ by {length_diff}"
        k_diff = bonds2[key][1] - bonds1[key][1]
        assert abs(k_diff) < 1e-9 * kj_nm2_mol, f"bond k differ by {k_diff}"


@requires_package("openmm")
def _compare_angle_forces(force1, force2):
    assert force1.getNumAngles() == force2.getNumAngles()

    angles1 = _create_angle_dict(force1)
    angles2 = _create_angle_dict(force2)

    for key in angles1:
        angle_diff = angles2[key][0] - angles1[key][0]
        assert abs(angle_diff) < 1e-15 * openmm.unit.radian, f"angles differ by {angle_diff}"
        k_diff = angles2[key][1] - angles1[key][1]
        assert abs(k_diff) < 1e-10 * kj_rad2_mol, f"angle k differ by {k_diff}"


def _compare_nonbonded_settings(force1, force2):
    for attr in dir(force1):
        if not attr.startswith("get") or attr in [
            "getExceptionParameterOffset",
            "getExceptionParameters",
            "getGlobalParameterDefaultValue",
            "getGlobalParameterName",
            "getLJPMEParametersInContext",
            "getPMEParametersInContext",
            "getParticleParameterOffset",
            "getParticleParameters",
            "getForceGroup",
        ]:
            continue
        assert getattr(force1, attr)() == getattr(force2, attr)(), attr


@requires_package("openmm")
def _compare_nonbonded_parameters(force1, force2):
    assert force1.getNumParticles() == force2.getNumParticles(), "found different number of particles"

    for i in range(force1.getNumParticles()):
        q1, sig1, eps1 = force1.getParticleParameters(i)
        q2, sig2, eps2 = force2.getParticleParameters(i)
        assert abs(q2 - q1) < 1e-8 * openmm.unit.elementary_charge, f"charge mismatch in particle {i}: {q1} vs {q2}"
        assert abs(sig2 - sig1) < 1e-12 * openmm.unit.nanometer, f"sigma mismatch in particle {i}: {sig1} vs {sig2}"
        assert (
            abs(eps2 - eps1) < 1e-12 * openmm.unit.kilojoule_per_mole
        ), f"epsilon mismatch in particle {i}: {eps1} vs {eps2}"


@requires_package("openmm")
def _compare_exceptions(force1, force2):
    assert force1.getNumExceptions() == force2.getNumExceptions(), "found different number of exceptions"

    for i in range(force1.getNumExceptions()):
        _, _, q1, sig1, eps1 = force1.getExceptionParameters(i)
        _, _, q2, sig2, eps2 = force2.getExceptionParameters(i)
        assert abs(q2 - q1) < 1e-12 * openmm.unit.elementary_charge**2, f"charge mismatch in exception {i}"
        assert abs(sig2 - sig1) < 1e-12 * openmm.unit.nanometer, f"sigma mismatch in exception {i}"
        assert abs(eps2 - eps1) < 1e-12 * openmm.unit.kilojoule_per_mole, f"epsilon mismatch in exception {i}"


@requires_package("openmm")
def _get_force(openmm_sys: "openmm.System", force_type):
    forces = [f for f in openmm_sys.getForces() if type(f) is force_type]

    if len(forces) > 1:
        raise NotImplementedError("Not yet able to process duplicate forces types")
    return forces[0]
