"""Custom warnings used in Interchange."""


class InterchangeDeprecationWarning(UserWarning):
    """Warning for deprecated portions of the Interchange API."""


class SwitchingFunctionNotImplementedWarning(UserWarning):
    """Exporting to an engine that does not implement a switching function."""
