class SMIRNOFFHandlerNotImplementedError(Exception):
    """
    Exception for when of the SMIRNOFF specification that are not implemented
    here are found
    """

    def __init__(self, *args):
        if args:
            self.name = args[0]

    def __str__(self):
        msg = "SMIRNOFF parameter not implemented here. "
        if self.name:
            msg += f"Tried to parse ParameterHandler of name {self.name}"
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


class JAXNotInstalledError(ImportError):
    """
    Exception for when JAX is called, but not installed
    """

    def __str__(self):
        msg = (
            "\nThis function requires JAX, which was not found to be installed."
            "\nInstall it with `conda install jax -c conda-forge` or"
            "\n`pip install --upgrade pip && pip install --upgrade jax jaxlib`."
        )
        return msg


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
