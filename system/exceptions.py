class SMIRNOFFHandlerNotImplementedError(Exception):
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        msg = f"SMIRNOFF parameter not implemented here. "
        if self.name:
            msg += "Tried to parse ParameterHandler of name {self.name}"
        return msg
