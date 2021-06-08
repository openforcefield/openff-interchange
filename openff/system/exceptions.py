class SMIRNOFFHandlersNotImplementedError(Exception):
    """
    Exception for when some parameter handlers in the SMIRNOFF specification
    are not implemented here
    """

    def __init__(self, *args):
        if args:
            if isinstance(args[0], str):
                self.names = [args[0]]
            elif isinstance(args[0], list):
                self.names = args[0]

    def __str__(self):
        msg = "SMIRNOFF parameters not implemented here: "
        for name in self.names:
            msg += f"\t{name}"
        return msg


class ToolkitTopologyConformersNotFoundError(Exception):
    """
    Exception for when reference molecules in a toolkit topology is required to contain
    conformers, but they are not found
    """

    def __init__(self, *args):
        if args:
            self.mol = str(args[0])

    def __str__(self):
        msg = "A reference molecule in the topology does not contain any conformers"
        if self.mol:
            msg += f"The molecule lacking a conformer is {self.mol}"


class InvalidParameterHandlerError(ValueError):
    """
    Generic exception for mismatch between expected and found ParameterHandler types
    """


class InvalidBoxError(ValueError):
    """
    Generic exception for errors reading box data
    """


class InvalidTopologyError(ValueError):
    """
    Generic exception for errors reading chemical topology data
    """


class InterMolEnergyComparisonError(AssertionError):
    """
    Exception for when energies derived from InterMol do not match
    """


class NonbondedEnergyError(AssertionError):
    """
    Exception for when non-bonded energies computed from different objects differ
    """


class InvalidExpressionError(ValueError):
    """
    Exception for when an expression cannot safely be interpreted
    """


class UnsupportedCutoffMethodError(BaseException):
    """
    Exception for a cutoff method that is invalid or not supported by an engine
    """


class UnimplementedCutoffMethodError(BaseException):
    """
    Exception for a cutoff method that should be supported but it not yet implemented
    """


class UnsupportedParameterError(ValueError):
    """
    Exception for parameters having unsupported values, i.e. non-1.0 idivf
    """


class UnsupportedBoxError(ValueError):
    """
    Exception for processing an unsupported box, probably non-orthogonal
    """


class UnsupportedExportError(BaseException):
    """
    Exception for attempting to write to an unsupported file format
    """

    def __init__(self, *args):
        if args:
            self.file_ext = args[0]

    def __str__(self):
        if self.file_ext:
            msg = f"Writing file format {self.file_ext} not supported."
        else:
            msg = "Writing unknown file format"
        return msg


class MissingBoxError(BaseException):
    """
    Exception for when box vectors are needed but missing
    """


class MissingPositionsError(BaseException):
    """
    Exception for when positions are needed but missing
    """


class MissingParametersError(BaseException):
    """
    Exception for when parameters are needed but missing
    """


class MissingUnitError(ValueError):
    """
    Exception for data missing a unit tag
    """


class UnitValidationError(ValueError):
    """
    Exception for bad behavior when validating unit-tagged data
    """


class NonbondedCompatibilityError(BaseException):
    """
    Exception for unsupported combination of nonbonded methods
    """


class MissingNonbondedCompatibilityError(BaseException):
    """
    Exception for uncovered combination of nonbonded methods
    """


class InternalInconsistencyError(BaseException):
    """
    Fallback exception for bad behavior. These should not be reached but
    are raised to safeguard against problematic edge cases silently passing.
    """


class SanderError(BaseException):
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
