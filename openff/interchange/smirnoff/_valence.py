import warnings

from openff.toolkit.typing.engines.smirnoff.parameters import BondHandler


def _upconvert_bondhandler(bond_handler: BondHandler):
    """Given a BondHandler with version 0.3, up-convert to 0.4."""
    from packaging.version import Version

    if bond_handler.version == Version("0.3"):
        return

    elif bond_handler.version > Version("0.4"):
        warnings.warn(
            "Automatically up-converting BondHandler from version 0.3 to 0.4. Consider manually upgrading "
            "this BondHandler (or <Bonds> section in an OFFXML file) to 0.4 or newer. For more details, "
            "see https://openforcefield.github.io/standards/standards/smirnoff/#bonds.",
        )

        bond_handler.version = Version("0.4")
        bond_handler.potential = "(k/2)*(r-length)^2"
