"""
Collections for custom SMIRNOFF handlers.
"""
from typing import List


def load_plugins() -> List:
    """Load external potential handlers as plugins."""
    from importlib_metadata import entry_points

    plugin_handlers = list()

    for entry_point in entry_points().select(
        group="openff.interchange.plugins.handlers",
    ):
        handler = entry_point.load()

        # if not issubclass(handler, SMIRNOFFPotentialHandler):
        # raise Exception(f"Handler {handler} and entry point {entry_point} not valid.")

        plugin_handlers.append(handler)

    return plugin_handlers
