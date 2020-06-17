class SMIRNOFFHandlerNotImplementedError(Exception):
    def __init__(self, args):
        if args:
            self.name = args[0]

    def __str__(self):
        msg = "SMIRNOFF parameter not implemented here. "
        if self.name:
            msg += f"Tried to parse ParameterHandler of name {self.name}"
        return msg


class JAXNotInstalledError(ImportError):
    def __str__(self):
        msg = (
            '\nThis function requires JAX, which was not found to be installed.'
            '\nInstall it with `conda install jax -c conda-forge` or'
            '\n`pip install --upgrade pip && pip install --upgrade jax jaxlib`.'
        )
        return msg
