from openff.toolkit import Quantity
from pydantic import Field

from openff.interchange._annotations import _Quantity
from openff.interchange.pydantic import _BaseModel


class Person(_BaseModel):
    mass: _Quantity = Field()


class Roster(_BaseModel):
    people: dict[str, Person] = Field(dict())

    foo: _Quantity = Field()


class Model(_BaseModel):
    array: _Quantity


def test_simple_model_validation():
    bob = Person(mass="100.0 kilogram")

    assert Person.model_validate(bob) == bob
    assert Person.model_validate(bob.model_dump()) == bob

    assert Person.model_validate_json(bob.model_dump_json()) == bob


def test_simple_model_setter():
    bob = Person(mass="100.0 kilogram")

    bob.mass = "90.0 kilogram"

    assert bob.mass == Quantity("90.0 kilogram")


def test_model_with_array_quantity():
    model = Model(array=Quantity([1, 2, 3], "angstrom"))

    for test_model in [
        Model.model_validate(model),
        Model.model_validate(model.model_dump()),
        Model.model_validate_json(model.model_dump_json()),
    ]:
        assert all(test_model.array == model.array)


def test_nested_model():
    roster = Roster(
        people={"Bob": {"mass": "100.0 kilogram"}, "Alice": {"mass": "70.0 kilogram"}},
        foo="10.0 year",
    )

    assert Roster.model_validate(roster) == roster
    assert Roster.model_validate(roster.model_dump()) == roster

    assert Roster.model_validate_json(roster.model_dump_json()) == roster
