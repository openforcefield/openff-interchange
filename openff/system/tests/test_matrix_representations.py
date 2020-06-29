import pytest
import numpy as np

from .base_test import BaseTest
from ..system import System
from ..utils import jax_available

class TestMatrixRepresentations(BaseTest):

    @pytest.mark.skipif(not jax_available, reason='Requires JAX')
    def test_to_p_q(self, argon_ff, argon_top):
        import jax
        import jax.numpy as jnp
        from jax.interpreters.xla import DeviceArray

        argon = System(
                topology=argon_top,
                forcefield=argon_ff,
                positions=np.zeros((argon_top.n_topology_atoms, 3)),
                box=np.array([10, 10, 10]),
            )

        # Probably better to get the SMIRNOFFvdWTerm directly from processing the force field
        term = argon.term_collection.terms['vdW']

        (p, mapping) = term.get_p(use_jax=True)

        assert isinstance(p, DeviceArray)
        assert p.shape == (1, 2)

        q = term.get_q(use_jax=True)

        assert isinstance(q, DeviceArray)
        assert q.shape == (4, 2)

        assert jnp.allclose(q, term.parametrize(p))

        param_matrix = term.get_param_matrix()

        ref = jnp.array(4 * [1.0, 0.0, 0.0, 1.0], ).reshape((8, 2))

        assert jnp.allclose(ref, param_matrix)
