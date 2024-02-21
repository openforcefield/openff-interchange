import random

import pytest
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    ParameterHandler,
)

from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff import (
    SMIRNOFFAngleCollection,
    SMIRNOFFCollection,
    SMIRNOFFElectrostaticsCollection,
)


class TestSMIRNOFFCollection:
    def test_allowed_parameter_handler_types(self):
        class DummyParameterHandler(ParameterHandler):
            pass

        class DummySMIRNOFFCollection(SMIRNOFFCollection):
            type = "Bonds"
            expression = "1+1"

            @classmethod
            def allowed_parameter_handlers(cls):
                return [DummyParameterHandler]

            @classmethod
            def supported_parameters(cls):
                return list()

        dummy_handler = DummySMIRNOFFCollection()
        angle_Handler = AngleHandler(version=0.3)

        assert DummyParameterHandler in dummy_handler.allowed_parameter_handlers()
        assert AngleHandler not in dummy_handler.allowed_parameter_handlers()
        assert (
            DummyParameterHandler
            not in SMIRNOFFAngleCollection.allowed_parameter_handlers()
        )

        dummy_handler = DummyParameterHandler(version=0.3)

        with pytest.raises(InvalidParameterHandlerError):
            SMIRNOFFAngleCollection.create(
                parameter_handler=dummy_handler,
                topology=Topology(),
            )

        with pytest.raises(InvalidParameterHandlerError):
            DummySMIRNOFFCollection.create(
                parameter_handler=angle_Handler,
                topology=Topology(),
            )


def test_json_roundtrip_preserves_float_values():
    """Reproduce issue #908."""
    scale_factor = 0.5 + random.random() * 0.5

    collection = SMIRNOFFElectrostaticsCollection(scale_14=scale_factor)

    assert collection.scale_14 == scale_factor

    roundtripped = SMIRNOFFElectrostaticsCollection.parse_raw(collection.json())

    assert roundtripped.scale_14 == scale_factor
