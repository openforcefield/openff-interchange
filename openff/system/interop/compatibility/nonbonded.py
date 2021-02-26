from openff.system.exceptions import (
    MissingNonbondedCompatibilityError,
    NonbondedCompatibilityError,
)

ALLOWED = [
    {
        "electrostatics_method": "PME",
        "vdw_method": "cutoff",
        "periodic_topology": True,
    },
    {
        "electrostatics_method": "PME",
        "vdw_method": "cutoff",
        "periodic_topology": False,
    },
    {
        "electrostatics_method": "cutoff",
        "vdw_method": "cutoff",
        "periodic_topology": True,
    },
]

DISALLOWED = [
    {
        "electrostatics_method": "Coulomb",
        "vdw_method": "cutoff",
        "periodic_topology": False,
    },
]


def check_nonbonded_compatibility(methods):
    """Check nonbonded methods against known allowed and disallowed
    combinations of nonbonded methods"""
    if methods["electrostatics_method"] in {"reaction-field"}:
        raise NonbondedCompatibilityError(
            "Electrostatics method reaction-field is not supported"
        )
    if methods in ALLOWED:
        return
    elif methods in DISALLOWED:
        raise NonbondedCompatibilityError(methods)
    else:
        raise MissingNonbondedCompatibilityError(methods)
