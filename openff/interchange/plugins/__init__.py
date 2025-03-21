"""
Collections for custom SMIRNOFF handlers.
"""

from openff.interchange.smirnoff import SMIRNOFFCollection

__all__ = [
    "SMIRNOFFCollection",
]


def load_smirnoff_plugins() -> list:
    """Load external potential handlers as plugins."""
    from importlib_metadata import entry_points
    from pydantic import PydanticUserError

    plugin_handlers = list()

    for entry_point in entry_points().select(
        group="openff.interchange.plugins.collections",
    ):
        try:
            plugin_handlers.append(entry_point.load())
        except PydanticUserError:
            continue

    return plugin_handlers
