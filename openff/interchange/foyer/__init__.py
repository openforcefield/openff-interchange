"""Foyer compatibility."""

# openff/interchange/foyer.py causes "import foyer" to not crash, which breaks has_packge("foyer")
# and breaks @skip_if_missing("foyer"), so need a more specific way of checking Foyer installation
try:
    from foyer import Forcefield

    has_foyer = True
except (ModuleNotFoundError, AttributeError):
    has_foyer = False
