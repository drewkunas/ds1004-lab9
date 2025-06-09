"""Minimal JAX wrapper for IPOPT with implicit differentiation support.

This module exposes a :func:`solve` function that can be used inside JAX
programs to solve small scale convex optimisation problems with optional
inequality constraints.  The solver is jittable and vmappable because the
heavy IPOPT call is executed through a host callback.  A custom VJP
implements implicit differentiation so that gradients of the solution with
respect to problem parameters can be obtained in a differentiable program.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

try:  # optional dependency
    import cyipopt  # type: ignore
except Exception:  # pragma: no cover
    cyipopt = None


@dataclass
class Problem:
    """Description of an optimisation problem.

    Parameters
    ----------
    fun:
        Objective function ``f(x, params)`` returning a scalar.
    x0:
        Initial primal variable guess.
    params:
        Arbitrary auxiliary parameters passed to ``fun`` and constraint
        functions.
    ineq_fun:
        Optional constraint function ``g(x, params)`` defining the
        elementwise inequality ``g(x, params) <= 0``.
    lb, ub:
        Optional lower and upper bounds on ``x``.
    """

    fun: Callable[[jnp.ndarray, Any], jnp.ndarray]
    x0: jnp.ndarray
    params: Any = None
    ineq_fun: Optional[Callable[[jnp.ndarray, Any], jnp.ndarray]] = None
    lb: Optional[jnp.ndarray] = None
    ub: Optional[jnp.ndarray] = None

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the objective with respect to ``x``."""
        return jax.grad(lambda y: self.fun(y, self.params))(x)

    def hess(self, x: jnp.ndarray) -> jnp.ndarray:
        """Hessian of the objective with respect to ``x``."""
        return jax.hessian(lambda y: self.fun(y, self.params))(x)

    def ineq(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate inequality constraints at ``x``.

        Returns an empty array if no inequality function was supplied."""
        if self.ineq_fun is None:
            return jnp.zeros(0, dtype=x.dtype)
        return self.ineq_fun(x, self.params)

    def jac_ineq(self, x: jnp.ndarray) -> jnp.ndarray:
        """Jacobian of the inequality constraints with respect to ``x``."""
        if self.ineq_fun is None:
            return jnp.zeros((0, x.size), dtype=x.dtype)
        return jax.jacobian(lambda y: self.ineq_fun(y, self.params))(x)


# ---------------------------------------------------------------------
# IPOPT integration via host callback.
# ---------------------------------------------------------------------

def _solve_ipopt_python(prob: Problem) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the problem with IPOPT in Python space.

    Parameters
    ----------
    prob:
        Instance of :class:`Problem` describing the optimisation task.

    Returns
    -------
    tuple of ``(x_opt, lam)``
        The primal optimum and the associated inequality multipliers.
    """

    if cyipopt is None:
        raise ImportError("cyipopt package is required but not installed")

    n = prob.x0.size
    m = int(prob.ineq(jnp.array(prob.x0)).size)

    lb = prob.lb if prob.lb is not None else jnp.full(n, -jnp.inf)
    ub = prob.ub if prob.ub is not None else jnp.full(n, jnp.inf)

    def objective(x):
        return float(prob.fun(jnp.array(x), prob.params))

    def gradient(x):
        return jnp.asarray(prob.grad(jnp.array(x)))

    def constraints(x):
        return jnp.asarray(prob.ineq(jnp.array(x)))

    def jacobian(x):
        return jnp.asarray(prob.jac_ineq(jnp.array(x))).ravel()

    solver = cyipopt.Problem(n=n, m=m, lb=lb, ub=ub,
                             cl=jnp.full(m, -jnp.inf), cu=jnp.zeros(m))
    solver.addOption("print_level", 0)
    solver.setObjective(objective)
    solver.setGradient(gradient)
    if m:
        solver.setConstraints(constraints)
        solver.setJacobian(jacobian)
    result = solver.solve(prob.x0)
    x_opt = result[0]
    info = result[1]
    lam = info.get("mult_g", jnp.zeros(m)) if m else jnp.zeros(0)
    return jnp.asarray(x_opt), jnp.asarray(lam)


def _solve_ipopt_jax(prob: Problem) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute IPOPT via a host callback inside a JAX transformation."""
    m = int(prob.ineq(prob.x0).size)
    result_shape = (
        jax.ShapeDtypeStruct(prob.x0.shape, prob.x0.dtype),
        jax.ShapeDtypeStruct((m,), prob.x0.dtype),
    )
    return jax.experimental.host_callback.call(_solve_ipopt_python, prob, result_shape)


# ---------------------------------------------------------------------
# Public API with implicit differentiation using a custom VJP.
# ---------------------------------------------------------------------

@jax.custom_vjp
def solve(
    fun: Callable[[jnp.ndarray, Any], jnp.ndarray],
    x0: jnp.ndarray,
    params: Any,
    ineq_fun: Optional[Callable[[jnp.ndarray, Any], jnp.ndarray]] = None,
    lb: Optional[jnp.ndarray] = None,
    ub: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Solve a parameterised convex optimisation problem.

    Parameters
    ----------
    fun, x0, params:
        See :class:`Problem`.
    ineq_fun:
        Optional inequality constraints ``g(x, params) <= 0``.
    lb, ub:
        Optional variable bounds passed directly to IPOPT.
    """

    prob = Problem(fun, x0, params, ineq_fun, lb, ub)
    x_opt, _ = _solve_ipopt_jax(prob)
    return x_opt


def _solve_fwd(fun, x0, params, ineq_fun=None, lb=None, ub=None):
    """Forward pass returning the primal optimum and auxiliary data."""
    prob = Problem(fun, x0, params, ineq_fun, lb, ub)
    x_star, lam = _solve_ipopt_jax(prob)
    return x_star, (x_star, lam, fun, ineq_fun, params)


def _solve_bwd(res, g):
    """Backward pass implementing implicit differentiation."""
    x_star, lam, fun, ineq_fun, params = res
    hess = jax.hessian(lambda x: fun(x, params))(x_star)
    grad_params = jax.jacobian(lambda p: jax.grad(fun)(x_star, p))(params)

    if ineq_fun is not None and lam.size:
        jac_g = jax.jacobian(lambda x: ineq_fun(x, params))(x_star)  # (m, n)
        active = lam > 1e-8
        J = jac_g[active]
        lam_act = lam[active]

        if lam_act.size:
            def weighted_constr(x):
                return jnp.sum(lam_act * ineq_fun(x, params)[active])

            hess += jax.hessian(weighted_constr)(x_star)
            dcdp = jax.jacobian(lambda p: ineq_fun(x_star, p))(params)[active]

            KKT = jnp.block([[hess, J.T], [J, jnp.zeros((J.shape[0], J.shape[0]))]])
            rhs = jnp.concatenate([-grad_params, -dcdp], axis=0)
            sol = jnp.linalg.solve(KKT, rhs)
            dxdp = sol[: x_star.size]
        else:
            dxdp = -jnp.linalg.solve(hess, grad_params)
    else:
        dxdp = -jnp.linalg.solve(hess, grad_params)

    grad_fun = None
    grad_x0 = jnp.zeros_like(x_star)
    grad_params_out = jnp.tensordot(g, dxdp, axes=1)
    return grad_fun, grad_x0, grad_params_out, None, None, None


solve.defvjp(_solve_fwd, _solve_bwd)

__all__ = ["solve", "Problem"]
