"""optionlab: option pricing, validation, hedging experiments and an ML surrogate.

`optionlab.surrogate` (PyTorch) is imported lazily so the numerical core
only needs NumPy and SciPy.
"""
from . import binomial, black_scholes, greeks, hedging, monte_carlo  # noqa: F401

__all__ = ["black_scholes", "binomial", "monte_carlo", "greeks", "hedging"]
