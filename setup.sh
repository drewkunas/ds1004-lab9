#!/bin/sh
# Install dependencies for jax_ipopt tests
pip install --upgrade pip
pip install "jax[cpu]" cyipopt
