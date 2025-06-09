"""JAX wrapper around IPOPT via the cyipopt package."""

from .solver import Problem, solve

__all__ = ["Problem", "solve"]
