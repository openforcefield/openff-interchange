from typing import TYPE_CHECKING

from openff.utilities.utilities import has_package, requires_package

if TYPE_CHECKING or has_package("openmm"):
    import openmm


@requires_package("openmm")
def get_14_scaling_factors(omm_sys: "openmm.System") -> tuple[list, list]:
    """Find the 1-4 scaling factors as they are applied to an OpenMM System."""
    nonbond_force = [
        f for f in omm_sys.getForces() if type(f) is openmm.NonbondedForce
    ][0]

    vdw_14 = list()
    coul_14 = list()

    for exception_idx in range(nonbond_force.getNumExceptions()):
        i, j, q, sig, eps = nonbond_force.getExceptionParameters(exception_idx)

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
