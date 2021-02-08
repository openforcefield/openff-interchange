ALLOWED = [
    {
        "electrostatics_method": "PME",
        "vdw_method": "cutoff",
        "periodic_topology": True,
    },
]

DISALLOWED = [
    {
        "electrostatics_method": "PME",
        "vdw_method": "cutoff",
        "periodic_topology": False,
    },
]


def check_nonbonded_compatibility(methods):
    """Check nonbonded methods against known allowed and disallowed
    combinations of nonbonded methods"""
    if methods["electrostatics_method"] in {"reaction-field", "Coluomb"}:
        raise NotImplementedError("Electrostatics method not supported")
    if methods in ALLOWED:
        return True
    elif methods in DISALLOWED:
        return False
    else:
        raise Exception
