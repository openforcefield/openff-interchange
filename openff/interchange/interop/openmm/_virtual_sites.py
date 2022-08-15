from typing import TYPE_CHECKING

from openff.interchange.exceptions import UnsupportedExportError

if TYPE_CHECKING:
    from openff.interchange.components.smirnoff import SMIRNOFFVirtualSiteHandler


def _check_virtual_site_exclusion_policy(handler: "SMIRNOFFVirtualSiteHandler"):
    _SUPPORTED_EXCLUSION_POLICIES = ("parents",)

    if handler.exclusion_policy not in _SUPPORTED_EXCLUSION_POLICIES:
        raise UnsupportedExportError(
            f"Found unsupported exclusion policy {handler.exclusion_policy}. "
            f"Supported exclusion policies are {_SUPPORTED_EXCLUSION_POLICIES}"
        )
