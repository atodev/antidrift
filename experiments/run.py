"""Synthetic experiment: does the intent Hamiltonian detect drift better than a memoryless baseline?

Usage: uv run python experiments/run.py [--embedder st|hash] [--seed N]
Writes experiments/results_<embedder>_<seed>.json.

Every method is scored per step. Its threshold is a quantile of the peak score over
benign calibration traces (95th percentile, and the maximum); it is then applied to
traces not used for calibration. Two splits are reported: an alternating half split,
and leave-one-task-out.
"""

import argparse
import json
import random
from pathlib import Path
from statistics import median

import numpy as np

from corpus import DISTRACTORS, TASKS
from hamiltonian import HashingEmbedder, IntentMonitor, SentenceTransformerEmbedder

STEPS = 30
SKIP = 5  # steps ignored at the start of every trace while the smoothed state settles
N_PER_FAMILY = 300
SEED = 20261013  # reporting seed; design choices were made on seed 1
FAMILIES = ["on_task", "excursion", "gradual", "abrupt"]
BENIGN, DRIFT = ("on_task", "excursion"), ("gradual", "abrupt")
METHODS = ["pointwise", "V_smoothed", "H", "pointwise_warmup_norm", "H_warmup_norm"]
QUANTILE = 0.95
PATIENCE = (1, 2, 3)


class CachedEmbedder:
    def __init__(self, inner):
        self._inner, self._cache = inner, {}

    def embed(self, text):
        if text not in self._cache:
            self._cache[text] = self._inner.embed(text)
        return self._cache[text]


def draw(rng, pool, prev):
    """Sample from pool, never repeating the previous step verbatim."""
    while (s := rng.choice(pool)) == prev:
        pass
    return s


def make_trace(rng, family):
    task = rng.choice(list(TASKS))
    on, exc = TASKS[task]["on_task"], TASKS[task]["excursion"]
    off = DISTRACTORS[rng.choice(list(DISTRACTORS))]
    onset = rng.randint(10, 16) if family in DRIFT else None
    exc_start = rng.randint(10, 20) if family == "excursion" else None
    steps, prev = [], None
    for t in range(STEPS):
        if family == "excursion" and exc_start <= t < exc_start + 3:
            pool = exc
        elif family == "abrupt" and t >= onset:
            pool = off
        elif family == "gradual" and t >= onset and rng.random() < min(1.0, (t - onset + 1) / 8):
            pool = off
        else:
            pool = on
        prev = draw(rng, pool, prev)
        steps.append(prev)
    return {"family": family, "task": task, "intent": TASKS[task]["intent"], "onset": onset, "steps": steps}


def score(trace, emb):
    """Per-step score for each method."""
    mon = IntentMonitor(trace["intent"], emb, e_ref=float("inf"))
    readings = [mon.observe(s) for s in trace["steps"]]
    d = [1.0 - float(mon.anchor @ emb.embed(s)) for s in trace["steps"]]
    T, V, H = ([getattr(r, a) for r in readings] for a in "TVH")
    norm = lambda xs: [x / median(xs[:SKIP]) for x in xs]
    return {"pointwise": d, "V_smoothed": V, "H": H, "pointwise_warmup_norm": norm(d), "H_warmup_norm": norm(H), "T": T}


def first_alarm(xs, tau, patience=1):
    run = 0
    for t in range(SKIP, len(xs)):
        run = run + 1 if xs[t] > tau else 0
        if run >= patience:
            return t
    return None


def peak(xs):
    return max(xs[SKIP:])


def auroc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def evaluate(items, calib_idx, test_idx, method, patience=1, quantile=QUANTILE):
    """items: list of (trace, scores). Returns counts by family for the test items."""
    tau = float(np.quantile([peak(items[i][1][method]) for i in calib_idx if items[i][0]["family"] in BENIGN], quantile))
    res = {"tau": round(tau, 4)}
    for f in FAMILIES:
        sub = [items[i] for i in test_idx if items[i][0]["family"] == f]
        alarms = [first_alarm(s[method], tau, patience) for _, s in sub]
        if f in BENIGN:
            res[f] = {"n": len(sub), "false_alarms": sum(a is not None for a in alarms)}
        else:
            hits = [a - t["onset"] for a, (t, _) in zip(alarms, sub) if a is not None and a >= t["onset"]]
            res[f] = {
                "n": len(sub),
                "detected": len(hits),
                "pre_onset_alarms": sum(a is not None and a < t["onset"] for a, (t, _) in zip(alarms, sub)),
                "median_delay": median(hits) if hits else None,
                "mean_delay": round(float(np.mean(hits)), 2) if hits else None,
            }
    if patience == 1:
        neg = [peak(s[method]) for i in test_idx if (t := items[i][0])["family"] in BENIGN for s in [items[i][1]]]
        for f in DRIFT:
            pos = [peak(items[i][1][method]) for i in test_idx if items[i][0]["family"] == f]
            res[f]["auroc_vs_benign"] = round(auroc(neg, pos), 4)
    return res


def merge(parts):
    """Sum counts across leave-one-task-out folds."""
    out = {}
    for f in FAMILIES:
        keys = ("n", "false_alarms") if f in BENIGN else ("n", "detected", "pre_onset_alarms")
        out[f] = {k: sum(p[f][k] for p in parts) for k in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["st", "hash"], default="st")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    emb = CachedEmbedder(SentenceTransformerEmbedder() if args.embedder == "st" else HashingEmbedder())
    rng = random.Random(args.seed)

    items = []
    for f in FAMILIES:
        for _ in range(N_PER_FAMILY):
            t = make_trace(rng, f)
            items.append((t, score(t, emb)))
    idx = list(range(len(items)))

    # Random split: even positions calibrate, odd positions test (families are balanced in both).
    calib, test = idx[0::2], idx[1::2]
    out = {
        "config": {"embedder": args.embedder, "seed": args.seed, "steps": STEPS, "skip": SKIP, "n_per_family": N_PER_FAMILY, "quantile": QUANTILE},
        "random_split": {m: evaluate(items, calib, test, m) for m in METHODS},
        "random_split_zero_calibration_alarms": {m: evaluate(items, calib, test, m, quantile=1.0) for m in METHODS},
        "random_split_patience": {m: {p: evaluate(items, calib, test, m, p) for p in PATIENCE} for m in ("pointwise", "H")},
        "leave_one_task_out": {},
    }
    for m in METHODS:
        folds = {}
        for held in TASKS:
            c = [i for i in idx if items[i][0]["task"] != held]
            t = [i for i in idx if items[i][0]["task"] == held]
            folds[held] = evaluate(items, c, t, m)
        pooled = merge(list(folds.values()))
        for f in DRIFT:
            pooled[f]["mean_fold_auroc"] = round(float(np.mean([v[f]["auroc_vs_benign"] for v in folds.values()])), 4)
        out["leave_one_task_out"][m] = {"pooled": pooled, "folds": folds}

    # Share of total energy that is kinetic, per family (how much the momentum term contributes).
    out["kinetic_share"] = {
        f: round(float(np.mean([sum(s["T"][SKIP:]) / sum(s["H"][SKIP:]) for t, s in items if t["family"] == f])), 4) for f in FAMILIES
    }
    # Mean energy by step, per family, for the figure (drift families aligned on onset).
    def mean_curve(f, key, align):
        rows = {}
        for t, s in items:
            if t["family"] != f:
                continue
            off = t["onset"] if align else 0
            for i, x in enumerate(s[key]):
                rows.setdefault(i - off, []).append(x)
        return {str(k): round(float(np.mean(v)), 4) for k, v in sorted(rows.items()) if len(v) >= N_PER_FAMILY // 2}
    out["mean_H"] = {f: mean_curve(f, "H", f in DRIFT) for f in FAMILIES}

    path = Path(__file__).parent / f"results_{args.embedder}_{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    brief = {k: v for k, v in out.items() if k not in ("mean_H",)}
    for m in METHODS:
        brief["leave_one_task_out"][m] = out["leave_one_task_out"][m]["pooled"]
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
