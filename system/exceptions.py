class SMIRNOFFHandlerNotImplementedError(Exception):
    def __init__(self, args):
        if args:
            self.name = args[0]

    def __str__(self):
        msg = "SMIRNOFF parameter not implemented here. "
        if self.name:
            msg += f"Tried to parse ParameterHandler of name {self.name}"
        return msg
