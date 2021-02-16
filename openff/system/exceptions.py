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


class MissingDependencyError(BaseException):
    """
    Exception for when an optional dependency is needed but not installed

    """

    def __init__(self, package_name):
        self.msg = (
            f"Missing dependency {package_name}. Try installing it "
            f"with\n\n$ conda install {package_name} -c conda-forge"
        )

        super().__init__(self.msg)


class InvalidBoxError(TypeError):
    """
    Generic exception for errors reading box data
    """


class InterMolEnergyComparisonError(AssertionError):
    """
    Exception for when energies derived from InterMol do not match
    """


class InvalidExpressionError(ValueError):
    """
    Exception for when an expression cannot safely be interpreted
    """


class UnsupportedCutoffMethodError(BaseException):
    """
    Exception for incompatibilities in cutoff methods
    """


class UnsupportedParameterError(ValueError):
    """
    Exception for parameters having unsupported values, i.e. non-1.0 idivf
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
