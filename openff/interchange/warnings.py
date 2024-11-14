"""Custom warnings used in Interchange."""


class InterchangeDeprecationWarning(UserWarning):
    """Warning for deprecated portions of the Interchange API."""


class SwitchingFunctionNotImplementedWarning(UserWarning):
    """Exporting to an engine that does not implement a switching function."""


class MissingPositionsWarning(UserWarning):
    """Warning for when positions are likely needed but missing."""


class NonbondedSettingsWarning(UserWarning):
    """Warning for when details of nonbonded implementations get messy."""


class ForceFieldModificationWarning(UserWarning):
    """Warning for when a ForceField is modified."""


class PresetChargesAndVirtualSitesWarning(UserWarning):
    """Warning when possibly using preset charges and virtual sites together."""


class InterchangeCombinationWarning(UserWarning):
    """Warning for when combining Interchange objects."""
