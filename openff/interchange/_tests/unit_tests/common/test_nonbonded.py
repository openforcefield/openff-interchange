import json

from openff.interchange.common._nonbonded import ElectrostaticsCollection


def test_properties_on_child_collections_serialized():
    blob = ElectrostaticsCollection(scale_14=2.1).model_dump_json()

    assert json.loads(blob)["scale_14"] == 2.1

    assert ElectrostaticsCollection.model_validate_json(blob).scale_14 == 2.1
