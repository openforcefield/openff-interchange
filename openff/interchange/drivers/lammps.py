"""Functions for running energy evluations with LAMMPS."""

import tempfile

import numpy
from openff.toolkit import Quantity
from openff.utilities import MissingOptionalDependencyError, requires_package

from openff.interchange import Interchange
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import LAMMPSNotFoundError, LAMMPSRunError


def get_lammps_energies(
    interchange: Interchange,
    round_positions: int | None = None,
    detailed: bool = False,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by LAMMPS.

    Parameters
    ----------
    interchange : openff.interchange.Interchange
        An OpenFF Interchange object to compute the single-point energy of
    round_positions : int, optional
        The number of decimal places, in nanometers, to round positions. This can be useful when
        comparing to i.e. GROMACS energies, in which positions may be rounded.
    detailed : bool, optional
        If True, return a detailed energy report containing all energy components.

    Returns
    -------
    report : EnergyReport
        An `EnergyReport` object containing the single-point energies.

    """
    try:
        return _process(
            _get_lammps_energies(interchange, round_positions),
            detailed,
        )
    except MissingOptionalDependencyError:
        raise LAMMPSNotFoundError


@requires_package("lammps")
def _get_lammps_energies(
    interchange: Interchange,
    round_positions: int | None = None,
) -> dict[str, Quantity]:
    import lammps

    if round_positions is not None:
        interchange.positions = numpy.round(
            interchange.positions,  # type: ignore[arg-type]
            round_positions,
        )

    with tempfile.TemporaryDirectory():
        interchange.to_lammps("out")

    # By default, LAMMPS spits out logs to the screen, turn it off
    # https://matsci.org/t/how-to-remove-or-redirect-python-lammps-stdout/38075/5
    # not that this is not sent to STDOUT, so `contextlib.redirect_stdout` won't work
    runner = lammps.lammps(cmdargs=["-screen", "none", "-nocite"])

    try:
        runner.file("out_pointenergy.in")
    # LAMMPS does not raise a custom exception :(
    except Exception as error:
        raise LAMMPSRunError from error

    # thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe
    parsed_energies = [Quantity(energy, "kilocalorie_per_mole") for energy in runner.last_thermo().values()]

    # TODO: Sanely map LAMMPS's energy names to the ones we care about
    return {
        "Bond": parsed_energies[0],
        "Angle": parsed_energies[1],
        "ProperTorsion": parsed_energies[2],
        "ImproperTorsion": parsed_energies[3],
        "vdW": parsed_energies[5],
        "DispersionCorrection": parsed_energies[8],
        "ElectrostaticsShort": parsed_energies[6],
        "ElectrostaticsLong": parsed_energies[7],
    }


def _process(
    energies: dict[str, Quantity],
    detailed: bool = False,
) -> EnergyReport:
    if detailed:
        return EnergyReport(energies=energies)

    return EnergyReport(
        energies={
            "Bond": energies["Bond"],
            "Angle": energies["Angle"],
            "Torsion": energies["ProperTorsion"] + energies["ImproperTorsion"],
            "vdW": energies["vdW"] + energies["DispersionCorrection"],
            "Electrostatics": (energies["ElectrostaticsShort"] + energies["ElectrostaticsLong"]),
        },
    )
