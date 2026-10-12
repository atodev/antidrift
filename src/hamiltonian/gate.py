"""Generation gate: a self-modification is accepted only if named invariants hold."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .embed import Embedder


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    deltas: dict[str, float]  # per-probe change, 1 - cos(reference, candidate)
    violated: list[str]


class GenerationGate:
    """Compares a candidate system's probe responses with those of generation zero.

    The reference is fixed at construction. Comparing each generation with its
    predecessor instead would let small accepted changes compound unseen.
    """

    def __init__(
        self,
        probes: dict[str, str],
        reference: Callable[[str], str],
        embedder: Embedder,
        tolerance: float = 0.15,
    ):
        self._probes = dict(probes)
        self._embedder = embedder
        self.tolerance = tolerance
        self._reference: dict[str, np.ndarray] = {}
        for name, prompt in self._probes.items():
            v = embedder.embed(reference(prompt))
            v.setflags(write=False)
            self._reference[name] = v

    def check(self, candidate: Callable[[str], str]) -> GateResult:
        deltas = {
            name: 1.0 - float(self._reference[name] @ self._embedder.embed(candidate(prompt)))
            for name, prompt in self._probes.items()
        }
        violated = [name for name, d in deltas.items() if d > self.tolerance]
        return GateResult(accepted=not violated, deltas=deltas, violated=violated)
