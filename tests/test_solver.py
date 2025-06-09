import jax
import jax.numpy as jnp
import numpy as np
import pytest

cyipopt = pytest.importorskip('cyipopt')

from jax_ipopt import solve

# simple quadratic problem f(x; b) = 0.5 * x^T x + b^T x

def fun(x, b):
    return 0.5 * jnp.dot(x, x) + jnp.dot(b, x)


def test_solve_quadratic():
    b = jnp.array([1.0, -1.0])
    x0 = jnp.zeros_like(b)
    x_star = solve(fun, x0, b)
    np.testing.assert_allclose(x_star, -b, rtol=1e-5, atol=1e-5)


def test_derivative_wrt_params():
    b = jnp.array([0.5, 0.25])
    x0 = jnp.zeros_like(b)
    jac_fun = jax.jacrev(lambda p: solve(fun, x0, p))
    jac = jac_fun(b)
    np.testing.assert_allclose(jac, -jnp.eye(2), rtol=1e-5, atol=1e-5)


def test_jit_and_vmap():
    bs = jnp.stack([jnp.array([1.0, 2.0]), jnp.array([-0.5, 0.0])])
    x0 = jnp.zeros(2)
    vmapped = jax.vmap(lambda p: solve(fun, x0, p))
    jitted = jax.jit(vmapped)
    result = jitted(bs)
    np.testing.assert_allclose(result, -bs, rtol=1e-5, atol=1e-5)


def test_with_inequality_constraints():
    b = jnp.array([0.0, 0.0])
    x0 = jnp.array([2.0, 2.0])

    def ineq(x, _):
        return x - 0.5

    x_star = solve(fun, x0, b, ineq_fun=ineq)
    np.testing.assert_array_less(x_star, jnp.array([0.5001, 0.5001]))
