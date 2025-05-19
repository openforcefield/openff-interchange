import packaging.version


def get_lammps_version() -> packaging.version.Version:
    """Convert LAMMPS's YYYYMMDD version string to a PEP 440-compliant version."""
    import lammps

    YYYYMMDD = str(lammps.lammps(cmdargs=["-screen", "none", "-nocite"]).version())

    YYYY = YYYYMMDD[:4]
    MM = YYYYMMDD[4:6]
    DD = YYYYMMDD[6:8]

    return packaging.version.Version(f"{YYYY}.{MM}.{DD}")
