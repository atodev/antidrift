"""Wrap any agent loop with the monitor. `agent_step` is whatever produces the next step."""

from collections.abc import Callable, Iterator

from hamiltonian import HashingEmbedder, IntentMonitor, Reading, Status


def guarded_loop(
    agent_step: Callable[[], tuple[str, str] | None],
    monitor: IntentMonitor,
    on_warn: Callable[[Reading], None] = lambda r: None,
) -> Iterator[Reading]:
    """Run the agent until it finishes or the monitor halts it.

    `agent_step` returns (port, text) for each step, or None when the agent is done.
    The check runs after every step and before the next one is allowed to start.
    """
    while (step := agent_step()) is not None:
        port, text = step
        reading = monitor.observe(text, port=port)
        yield reading
        if reading.status is Status.HALT:
            return
        if reading.status is Status.WARN:
            on_warn(reading)


if __name__ == "__main__":
    script = iter(
        [("planner", "Plan the migration of the billing database to PostgreSQL.")] * 6
        + [("search", "Top ten beaches in Sardinia for a summer holiday.")] * 6
    )
    monitor = IntentMonitor(
        "Migrate the billing database from MySQL to PostgreSQL.", HashingEmbedder()
    )
    for r in guarded_loop(lambda: next(script, None), monitor):
        print(f"{r.step:>2} {r.port:<8} H={r.H:.3f} V={r.V:.3f} {r.status.value}")
    print("energy by port:", monitor.ledger.injected())
