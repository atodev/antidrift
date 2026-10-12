"""The intent monitor: phase-space state, energy readings, and status rules."""

from dataclasses import dataclass
from enum import Enum
from statistics import median

import numpy as np

from .embed import Embedder
from .energy import kinetic, potential
from .ports import PortLedger


class Status(str, Enum):
    OK = "OK"
    WARN = "WARN"
    HALT = "HALT"


@dataclass(frozen=True)
class Reading:
    step: int
    T: float
    V: float
    H: float
    dH: float
    W: float | None  # H - E_ref; None until E_ref is known
    status: Status
    port: str | None = None


class IntentMonitor:
    """Tracks an agent trace as motion in a potential well centred on the intent.

    q_t is a smoothed, unit-norm embedding of the steps so far; p_t = m (q_t - q_{t-1}).
    H = |p|^2 / 2m + (k/2) |q - q*|^2, where q* is the frozen intent anchor.

    Pass `e_ref` from `calibrate_e_ref` where benign traces exist. Without it the
    monitor falls back to `margin` times the median H of its first `warmup` steps,
    which is a noisier reference. No alarm is raised during the first `settle` steps.
    """

    def __init__(
        self,
        intent: str | list[str],
        embedder: Embedder,
        *,
        m: float = 1.0,
        k: float = 1.0,
        alpha: float = 0.3,
        e_ref: float | None = None,
        warmup: int = 5,
        settle: int = 5,
        margin: float = 1.25,
        halt_factor: float = 1.0,
        patience: int = 2,
    ):
        texts = [intent] if isinstance(intent, str) else list(intent)
        anchor = np.mean([embedder.embed(t) for t in texts], axis=0)
        anchor = anchor / np.linalg.norm(anchor)
        anchor.setflags(write=False)
        self._anchor = anchor
        self._embedder = embedder
        self.m, self.k, self.alpha = m, k, alpha
        self.e_ref = e_ref
        self.warmup, self.settle, self.margin = warmup, settle, margin
        self.halt_factor, self.patience = halt_factor, patience
        self.ledger = PortLedger()
        self.readings: list[Reading] = []
        self._q: np.ndarray | None = None
        self._over = 0
        self._halted = False

    @property
    def anchor(self) -> np.ndarray:
        """The intent anchor q*. Read-only: computed once and never updated."""
        return self._anchor

    def observe(self, text: str, port: str | None = None) -> Reading:
        e = self._embedder.embed(text)
        if self._q is None:
            q = e
            p = np.zeros_like(e)
        else:
            q = (1.0 - self.alpha) * self._q + self.alpha * e
            q = q / np.linalg.norm(q)
            p = self.m * (q - self._q)
        self._q = q

        T = kinetic(p, self.m)
        V = potential(q, self._anchor, self.k)
        H = T + V
        dH = H - self.readings[-1].H if self.readings else 0.0

        if self.e_ref is None and len(self.readings) + 1 >= self.warmup:
            self.e_ref = self.margin * median([r.H for r in self.readings] + [H])

        reading = Reading(
            step=len(self.readings),
            T=T,
            V=V,
            H=H,
            dH=dH,
            W=None if self.e_ref is None else H - self.e_ref,
            status=self._status(V, H),
            port=port,
        )
        self.readings.append(reading)
        if port is not None:
            self.ledger.record(port, dH)
        return reading

    def _status(self, V: float, H: float) -> Status:
        if self._halted:
            return Status.HALT
        if self.e_ref is None or len(self.readings) < self.settle:
            return Status.OK  # no budget yet, or the smoothed state is still settling
        self._over = self._over + 1 if H > self.halt_factor * self.e_ref else 0
        if self._over >= self.patience:
            self._halted = True
            return Status.HALT
        return Status.WARN if V > self.e_ref else Status.OK


def calibrate_e_ref(
    runs: list[tuple[str | list[str], list[str]]],
    embedder: Embedder,
    quantile: float = 0.95,
    skip: int = 5,
    **monitor_kwargs,
) -> float:
    """Energy budget from benign runs: a quantile of each run's peak H after `skip` steps.

    `runs` is a list of (intent, steps) pairs known to have stayed on intent.
    """
    peaks = []
    for intent, steps in runs:
        mon = IntentMonitor(intent, embedder, e_ref=float("inf"), **monitor_kwargs)
        energies = [mon.observe(s).H for s in steps]
        peaks.append(max(energies[skip:] or energies))
    return float(np.quantile(peaks, quantile))
