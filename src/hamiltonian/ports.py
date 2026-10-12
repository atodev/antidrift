"""Per-port energy ledger: which input channel supplied the energy."""

from collections import defaultdict


class PortLedger:
    def __init__(self) -> None:
        self._injected: dict[str, float] = defaultdict(float)
        self._steps: dict[str, int] = defaultdict(int)

    def record(self, port: str, dH: float) -> None:
        self._injected[port] += dH
        self._steps[port] += 1

    def injected(self) -> dict[str, float]:
        """Net energy change attributed to each port, largest first."""
        return dict(sorted(self._injected.items(), key=lambda kv: kv[1], reverse=True))

    def steps(self) -> dict[str, int]:
        return dict(self._steps)
