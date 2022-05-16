"""Custom exceptions used in Interchange."""
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class SMIRNOFFParameterAttributeNotImplementedError(Exception):
    """
    Exception for when a parameter attribute is supported by the SMIRNOFF specification but not yet implemented.

    For example, this was raised when k_bondorder (used in bond order-based interpolation of force constants)
    before the behavior was supported.
    """


class SMIRNOFFHandlersNotImplementedError(Exception):
    """
    Exception for when some parameter handlers in the SMIRNOFF specification are not implemented here.
    """

    def __init__(self, *args: Union[List, str]) -> None:
        if args:
            if isinstance(args[0], str):
                self.names = [args[0]]
            elif isinstance(args[0], list):
                self.names = args[0]

    def __str__(self) -> str:
        msg = "SMIRNOFF parameters not implemented here: "
        for name in self.names:
            msg += f"\t{name}"
        return msg


class SMIRNOFFVersionNotSupportedError(Exception):
    """
    Exception for when a parameter handler's version is not supported.
    """


class ToolkitTopologyConformersNotFoundError(Exception):
    """
    Exception for when reference molecules in a toolkit topology lack conformers.
    """

    def __init__(self, *args: "Molecule") -> None:
        if args:
            self.mol = str(args[0])

    def __str__(self) -> str:
        msg = "A reference molecule in the topology does not contain any conformers"
        if self.mol:
            msg += f"The molecule lacking a conformer is {self.mol}"
        return msg


class InvalidParameterHandlerError(ValueError):
    """
    Generic exception for mismatch between expected and found ParameterHandler types.
    """


class InvalidBoxError(ValueError):
    """
    Generic exception for errors reading box data.
    """


class InvalidTopologyError(ValueError):
    """
    Generic exception for errors reading chemical topology data.
    """


class NonbondedEnergyError(AssertionError):
    """
    Exception for when non-bonded energies computed from different objects differ.
    """


class InvalidExpressionError(ValueError):
    """
    Exception for when an expression cannot safely be interpreted.
    """


class UnsupportedCutoffMethodError(BaseException):
    """
    Exception for a cutoff method that is invalid or not supported by an engine.
    """


class UnimplementedCutoffMethodError(BaseException):
    """
    Exception for a cutoff method that should be supported but it not yet implemented.
    """


class UnsupportedParameterError(ValueError):
    """
    Exception for parameters having unsupported values, i.e. non-1.0 idivf.
    """


class UnsupportedBoxError(ValueError):
    """
    Exception for processing an unsupported box, probably non-orthogonal.
    """


class UnsupportedExportError(BaseException):
    """
    Exception for attempting to write to an unsupported file format.
    """

    def __init__(self, *args: str) -> None:
        if args:
            self.file_ext = args[0]

    def __str__(self) -> str:
        if self.file_ext:
            msg = f"Writing file format {self.file_ext} not supported."
        else:
            msg = "Writing unknown file format"
        return msg


class UnsupportedCombinationError(BaseException):
    """General exception for something going wrong in Interchange object combination."""


class ConversionError(BaseException):
    """
    Base exception for error handling during object conversion.
    """


class MissingBoxError(BaseException):
    """
    Exception for when box vectors are needed but missing.
    """


class MissingPositionsError(BaseException):
    """
    Exception for when positions are needed but missing.
    """


class MissingParameterHandlerError(BaseException):
    """
    Exception for when a parameter handler is requested but not found.
    """


class MissingParametersError(BaseException):
    """
    Exception for when parameters are needed but missing.
    """


class MissingBondOrdersError(BaseException):
    """
    Exception for when a parameter handler needs fractional bond orders but they are missing.
    """


class MissingUnitError(ValueError):
    """
    Exception for data missing a unit tag.
    """


class UnitValidationError(ValueError):
    """
    Exception for bad behavior when validating unit-tagged data.
    """


class NonbondedCompatibilityError(BaseException):
    """
    Exception for unsupported combination of nonbonded methods.
    """


class MissingNonbondedCompatibilityError(BaseException):
    """
    Exception for uncovered combination of nonbonded methods.
    """


class InternalInconsistencyError(BaseException):
    """
    Fallback exception for bad behavior releating to a self-inconsistent internal state.

    These should not be reached but are raised to safeguard against problematic edge cases silently passing.
    """


class AmberError(BaseException):
    """
    Base exception for handling Amber-related errors.
    """


class AmberExecutableNotFoundError(AmberError):
    """
    Exception for when an Amber-related excutable is not found.
    """


class SanderError(AmberError):
    """
    Exception for when a sander subprocess fails.
    """


class GMXRunError(BaseException):
    """
    Exception for when a GROMACS subprocess fails.
    """


class GMXGromppError(GMXRunError):
    """
    Exception for when `gmx grompp` fails.
    """


class GMXMdrunError(GMXRunError):
    """
    Exception for when `gmx mdrun` fails.
    """


class LAMMPSRunError(BaseException):
    """
    Exception for when a LAMMPS subprocess fails.
    """


class EnergyError(BaseException):
    """
    Base class for energies in reports not matching.
    """


class MissingEnergyError(BaseException):
    """
    Exception for when one report has a value for an energy group but the other does not.
    """


class NonIntegralMoleculeChargeException(BaseException):
    """
    Exception raised when the partial charges on a molecule do not sum up to its formal charge.
    """
