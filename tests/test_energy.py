import numpy as np
import pytest

from hamiltonian import hamiltonian, kinetic, potential


def unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_energy_is_zero_at_rest_on_the_anchor():
    a = unit(1, 0, 0)
    assert hamiltonian(a, np.zeros(3), a, m=1.0, k=1.0) == 0.0


def test_potential_equals_k_times_one_minus_cosine_for_unit_vectors():
    a, q = unit(1, 0, 0), unit(1, 1, 0)
    assert potential(q, a, k=2.0) == pytest.approx(2.0 * (1.0 - float(q @ a)))


def test_potential_is_symmetric():
    a, q = unit(1, 2, 3), unit(3, 1, 0)
    assert potential(q, a, 1.0) == pytest.approx(potential(a, q, 1.0))


def test_kinetic_scales_inversely_with_mass():
    p = np.array([0.3, 0.4, 0.0])
    assert kinetic(p, m=1.0) == pytest.approx(0.125)
    assert kinetic(p, m=2.0) == pytest.approx(0.0625)
