import numpy as np
import pytest

from openff.system.tests.base_test import BaseTest
from openff.system.utils import get_test_file_path, jax_available

SUPPORTED_HANDLER_MAPPING = {}


class TestMatrixRepresentations(BaseTest):
    @pytest.mark.skipif(not jax_available, reason="Requires JAX")
    @pytest.mark.parametrize(
        "handler_name,n_ff_terms,n_sys_terms",
        [("vdW", 10, 72), ("Bonds", 8, 64), ("Angles", 6, 104)],
    )
    def test_to_force_field_to_system_parameters(
        self, parsley, ethanol_top, handler_name, n_ff_terms, n_sys_terms
    ):
        import jax.numpy as jnp
        from jax.interpreters.xla import DeviceArray

        from openff.system.stubs import ForceField

        handler = parsley[handler_name].create_potential(topology=ethanol_top)

        p = handler.get_force_field_parameters()

        assert isinstance(p, DeviceArray)
        assert np.prod(p.shape) == n_ff_terms

        q = handler.get_system_parameters()

        assert isinstance(q, DeviceArray)
        assert np.prod(q.shape) == n_sys_terms

        assert jnp.allclose(q, handler.parametrize(p))

        param_matrix = handler.get_param_matrix()

        ref_file = get_test_file_path(f"ethanol_param_{handler_name.lower()}.npy")
        ref = jnp.load(ref_file)

        assert jnp.allclose(ref, param_matrix)

        # TODO: Update with other handlers that can safely be assumed to follow 1:1 slot:smirks mapping
        if handler_name in ["vdW", "Bonds", "Angles"]:
            assert np.allclose(
                np.sum(param_matrix, axis=1), np.ones(param_matrix.shape[0])
            )
