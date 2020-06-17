import numpy as np
import jax
import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray

from .base_test import BaseTest
from ..system import System


class TestMatrixRepresentations(BaseTest):

    def test_to_p_q(self, argon_ff, argon_top):
        argon = System(
                topology=argon_top,
                forcefield=argon_ff,
                positions=np.zeros((argon_top.n_topology_atoms, 3)),
                box=np.array([10, 10, 10]),
            )

        # Probably better to get the SMIRNOFFvdWTerm directly from processing the force field
        term = argon.term_collection.terms['vdW']

        (p, mapping) = term.get_p()

        assert isinstance(p, (np.ndarray, jax.interpreters.xla.DeviceArray))
        assert p.shape == (1, 2)

        q = term.get_q()

        assert isinstance(q, (np.ndarray, jax.interpreters.xla.DeviceArray))
        assert q.shape == (4, 2)

        assert jnp.allclose(q, term.parametrize(p))

        param_matrix = term.get_param_matrix()

        ref = jnp.array(4 * [1.0, 0.0, 0.0, 1.0], ).reshape((8, 2))

        assert jnp.allclose(ref, param_matrix)
