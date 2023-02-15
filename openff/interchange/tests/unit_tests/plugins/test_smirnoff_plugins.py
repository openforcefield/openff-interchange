from nonbonded_plugins import SMIRNOFFBuckinghamCollection

from openff.interchange.plugins import load_smirnoff_plugins


def test_load_smirnoff_plugins():
    assert load_smirnoff_plugins() == [SMIRNOFFBuckinghamCollection]
