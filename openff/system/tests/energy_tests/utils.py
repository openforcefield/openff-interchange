from openff.system.tests.energy_tests.report import EnergyReport


def compare_gromacs_openmm(
    gmx_report: EnergyReport,
    omm_report: EnergyReport,
    # custom_tolerances: Dict[str, float] = None,
):

    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    gmx_report.compare(omm_report)


def compare_gromacs(
    report1: EnergyReport,
    report2: EnergyReport,
    #    custom_tolerances: Dict[str, float] = None,
):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    report1.compare(report2)
    # TODO: Fix constraints and other issues around GROMACS non-bonded energies


def compare_openmm(
    report1: EnergyReport,
    report2: EnergyReport,
    #    custom_tolerances: Dict[str, float] = None,
):

    report1.compare(report2)
