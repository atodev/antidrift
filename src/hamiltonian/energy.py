"""Energy terms of the intent Hamiltonian."""

import numpy as np


def kinetic(p: np.ndarray, m: float) -> float:
    """T = |p|^2 / 2m."""
    return float(p @ p) / (2.0 * m)


def potential(q: np.ndarray, anchor: np.ndarray, k: float) -> float:
    """V = (k/2) |q - q*|^2. For unit vectors this equals k (1 - cos(q, q*))."""
    d = q - anchor
    return 0.5 * k * float(d @ d)


def hamiltonian(q: np.ndarray, p: np.ndarray, anchor: np.ndarray, m: float, k: float) -> float:
    return kinetic(p, m) + potential(q, anchor, k)
