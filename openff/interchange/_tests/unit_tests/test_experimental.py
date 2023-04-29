import pytest

from openff.interchange._experimental import experimental
from openff.interchange.exceptions import ExperimentalFeatureException


def test_default():
    @experimental
    def f():
        pass

    with pytest.raises(ExperimentalFeatureException):
        f()


def test_experimental_opted_in(monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

    @experimental
    def g():
        pass

    g()


def test_experimental_opted_in_bad_value(monkeypatch):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "True")

    @experimental
    def h():
        pass

    with pytest.raises(ExperimentalFeatureException):
        h()