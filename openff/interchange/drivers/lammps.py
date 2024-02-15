"""Functions for running energy evluations with LAMMPS."""

import tempfile
from typing import Optional

import numpy
from openff.units import Quantity, unit
from openff.utilities import MissingOptionalDependencyError, requires_package

from openff.interchange import Interchange
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.drivers.report import EnergyReport
from openff.interchange.exceptions import LAMMPSNotFoundError, LAMMPSRunError


def get_lammps_energies(
    interchange: Interchange,
    round_positions: Optional[int] = None,
    detailed: bool = False,
) -> EnergyReport:
    """
    Given an OpenFF Interchange object, return single-point energies as computed by LAMMPS.

    .. warning :: This API is experimental and subject to change.

    .. todo :: Split out _running_ LAMMPS into a separate internal function

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
    round_positions: Optional[int] = None,
) -> dict[str, unit.Quantity]:
    import lammps

    if round_positions is not None:
        interchange.positions = numpy.round(interchange.positions, round_positions)

    with tempfile.TemporaryDirectory():
        interchange.to_lammps("out.lmp")
        mdconfig = MDConfig.from_interchange(interchange)
        mdconfig.write_lammps_input(
            interchange=interchange,
            input_file="tmp.in",
        )

    runner = lammps.lammps()

    try:
        runner.file("tmp.in")
    # LAMMPS does not raise a custom exception :(
    except Exception as error:
        raise LAMMPSRunError from error

    # thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe
    parsed_energies = [
        Quantity(energy, "kilocalorie_per_mole")
        for energy in runner.last_thermo().values()
    ]

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
    energies: dict[str, unit.Quantity],
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
            "Electrostatics": (
                energies["ElectrostaticsShort"] + energies["ElectrostaticsLong"]
            ),
        },
    )
