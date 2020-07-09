import pytest
import numpy as np

from .base_test import BaseTest
from ..typing.smirnoff.data import SUPPORTED_HANDLER_MAPPING
from ..utils import jax_available, get_test_file_path


class TestMatrixRepresentations(BaseTest):
    @pytest.mark.skipif(not jax_available, reason='Requires JAX')
    @pytest.mark.parametrize(
        'handler_name,n_ff_terms,n_sys_terms',
        [('vdW', 10, 36), ('Bonds', 8, 32), ('Angles', 6, 52)],
    )
    def test_to_force_field_to_system_parameters(
        self, parsley, ethanol_top, handler_name, n_ff_terms, n_sys_terms
    ):
        import jax.numpy as jnp
        from jax.interpreters.xla import DeviceArray

        term = SUPPORTED_HANDLER_MAPPING[handler_name].build_from_toolkit_data(
            handler=parsley[handler_name], topology=ethanol_top,
        )

        (p, mapping) = term.get_force_field_parameters(use_jax=True)

        assert isinstance(p, DeviceArray)
        assert p.shape == (n_ff_terms,)

        q = term.get_system_parameters(use_jax=True)

        assert isinstance(q, DeviceArray)
        assert q.shape == (n_sys_terms,)

        assert jnp.allclose(q, term.parametrize(p))

        param_matrix = term.get_param_matrix()

        ref = jnp.load(get_test_file_path(f'ethanol_param_{handler_name.lower()}.npy'))

        assert jnp.allclose(ref, param_matrix)

        # TODO: Update with other handlers that can safely be assumed to follow 1:1 slot:smirks mapping
        if handler_name in ['vdW', 'Bonds', 'Angles']:
            assert np.allclose(
                np.sum(param_matrix, axis=1), np.ones(param_matrix.shape[0])
            )
