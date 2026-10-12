# antidrift

A monitor for intent drift in long-running AI agents and self-improving systems.

It treats an agent's trace as motion in a potential well centred on the original intent.
Each step is embedded as a vector, a smoothed state tracks where the run currently sits,
and an energy function (the *intent Hamiltonian*) measures how far that state is from the
intent. Energy above a budget calibrated on benign runs is the drift signal.

This repository is the reference implementation for the whitepaper
*Intent as a Conserved Quantity: Hamiltonian Monitoring of Agentic and Recursively
Self-Improving Systems*.

## Paper

**[Read the whitepaper (PDF)](paper/intent-as-a-conserved-quantity.pdf)**

The [living document](https://claude.ai/code/artifact/a2cedb60-34e0-47e9-aed8-6c910892a808)
is the editable source; that link opens only for people it has been shared with.

## What the experiment found

The evaluation used 1,200 scripted traces across four families: on-task, legitimate
excursion, gradual drift and abrupt drift.

- **Memory helps.** At a threshold with no false alarms on benign traces, the monitor
  detected 300 of 300 drifting traces. A memoryless similarity check detected 267 of 300.
- **The kinetic term added nothing.** The full Hamiltonian and its potential term alone
  gave the same detections. What was tested is in effect a smoothed distance to the
  intent with a calibrated threshold.
- **The energy budget is task-specific.** A budget calibrated on two tasks raised many
  false alarms on one of three held-out tasks.
- **The evidence is synthetic.** Drift in the experiment is a scripted change of topic.
  Nothing here establishes performance on real agent workloads.

## What is in the box

| Component | File | Evaluated |
| --- | --- | --- |
| Energy kernel (T, V, H) | `src/hamiltonian/energy.py` | Yes |
| Intent monitor, status rules, budget calibration | `src/hamiltonian/monitor.py` | Yes |
| Per-port energy ledger (which input caused the drift) | `src/hamiltonian/ports.py` | Unit tests only |
| Generation gate for self-modification | `src/hamiltonian/gate.py` | Unit tests only |
| Embedders (sentence-transformers, offline hashing) | `src/hamiltonian/embed.py` | Yes |

The Python package is named `hamiltonian`.

## Quick start

```
uv venv
uv pip install -e ".[dev,st]"
```

The `st` extra installs sentence-transformers, which the experiment needs. The tests and
the example run without it.

```
uv run pytest                                 # 12 tests, offline
uv run python examples/agent_hook.py          # toy guarded agent loop
uv run python experiments/run.py              # reporting seed
uv run python experiments/run.py --seed 7919  # confirmation seed
uv run python experiments/run.py --seed 1     # development seed
```

Each experiment run writes `experiments/results_<embedder>_<seed>.json`. The results
files behind the paper's tables are committed.

## Using the monitor

```python
from hamiltonian import IntentMonitor, SentenceTransformerEmbedder, Status, calibrate_e_ref

embedder = SentenceTransformerEmbedder()
e_ref = calibrate_e_ref(benign_runs, embedder)   # list of (intent, steps) pairs

monitor = IntentMonitor("Migrate the billing database to PostgreSQL.", embedder, e_ref=e_ref)
for port, text in agent_steps:
    reading = monitor.observe(text, port=port)
    if reading.status is Status.HALT:
        print(monitor.ledger.injected())         # energy added, by input channel
        break
```

## Layout

- `src/hamiltonian/` — the library, 289 lines
- `tests/` — unit tests
- `examples/agent_hook.py` — wrapping an agent loop with the monitor
- `experiments/corpus.py` — step text for the synthetic experiment
- `experiments/run.py` — trace generation, scoring and evaluation
- `experiments/results_*.json` — results for the reporting, confirmation and development seeds
- `paper/` — the whitepaper as a PDF, and its licence

## Status

Research prototype. The paper, this code and the experiment's step text were drafted with
an AI assistant and have not been independently reviewed.

## Citing

If you use this software or the concept it implements, please cite the paper. GitHub's
"Cite this repository" button gives a ready-made reference from `CITATION.cff`.

> Tom Butler. *Intent as a Conserved Quantity: Hamiltonian Monitoring of Agentic and
> Recursively Self-Improving Systems.* 2026. https://github.com/atodev/antidrift

## Licence

- Code: [Apache-2.0](LICENSE). Redistributions must keep the [NOTICE](NOTICE) file.
- Paper: [CC BY 4.0](paper/LICENSE.md).

© 2026 Tom Butler
