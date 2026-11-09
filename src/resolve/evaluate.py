"""The evaluation: train the matcher on one half of the people, resolve the other half, grade everything."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
from sklearn.metrics import average_precision_score

from . import data, metrics, pipeline
from .features import FEATURES, Clean, PairFeatures, clean

THRESHOLD = 0.5


@dataclass
class Result:
    records: int
    people: int
    test_records: int
    test_people: int
    blocking: dict[str, float] = field(default_factory=dict)
    matcher: dict[str, float] = field(default_factory=dict)
    methods: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> metrics
    lookalikes: dict[str, dict[str, float]] = field(default_factory=dict)  # method -> kind -> wrongly merged share
    sensitivity: list[dict[str, float]] = field(default_factory=list)
    importance: list[tuple[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def _lookalike_pairs(rs: list[data.Record]) -> dict[str, list[tuple[int, int]]]:
    """Record pairs of *different* people in one household, by how the second relates to the first."""
    homes: dict[int, list[int]] = {}
    for i, r in enumerate(rs):
        homes.setdefault(r.household, []).append(i)
    out: dict[str, list[tuple[int, int]]] = {"twin": [], "junior": [], "family": []}
    for members in homes.values():
        for i, j in combinations(members, 2):
            if rs[i].person != rs[j].person:
                kind = rs[i].kind or rs[j].kind
                out[kind].append((i, j))
    return out


def run(n_people: int = 6000, seed: int = 7) -> Result:
    t0 = time.perf_counter()
    recs = data.generate(n_people, seed)
    people = sorted({r.person for r in recs})
    rng = np.random.default_rng(seed)
    train_people = set(rng.choice(people, len(people) // 2, replace=False).tolist())
    train = [r for r in recs if r.person in train_people]
    test = [r for r in recs if r.person not in train_people]
    r = Result(len(recs), len(people), len(test), len({x.person for x in test}))

    def prepare(rs: list[data.Record]) -> tuple[list[Clean], pipeline.Blocking, PairFeatures, np.ndarray, list[int]]:
        cs = [clean(x) for x in rs]
        b = pipeline.block(cs)
        pf = pipeline.pair_features(cs)
        return cs, b, pf, pf.matrix(b.pairs), [x.person for x in rs]

    _, tr_b, _, tr_x, tr_truth = prepare(train)
    tr_y = np.array([tr_truth[i] == tr_truth[j] for i, j in tr_b.pairs])
    matcher = pipeline.Matcher().fit(tr_x, tr_y)

    cs, b, pf, x, truth = prepare(test)
    y = np.array([truth[i] == truth[j] for i, j in b.pairs])
    tp = metrics.true_pairs(truth)
    n = len(cs)
    r.blocking = {
        "candidate_pairs": len(b.pairs),
        "all_pairs": n * (n - 1) / 2,
        "pair_completeness": len(tp & set(b.pairs)) / len(tp),
        "reduction_ratio": 1 - len(b.pairs) / (n * (n - 1) / 2),
        "skipped_blocks": b.skipped_blocks,
    }
    prob = matcher.prob(x)
    r.matcher = {"average_precision": float(average_precision_score(y, prob)), "positives": float(y.mean())}

    def score(i: int, j: int) -> float:
        return float(matcher.prob(pf.matrix([(i, j)]))[0])

    results = {
        "exact match (email, or name + birth date)": pipeline.exact_match_baseline(cs),
        "hand-written rules": pipeline.rules_baseline(cs, b.pairs),
        "matcher + connected components": pipeline.connected_components(n, b.pairs, prob, THRESHOLD),
        "matcher + average linkage": pipeline.average_linkage(n, b.pairs, prob, THRESHOLD, score),
    }
    look = _lookalike_pairs(test)
    for name, labels in results.items():
        pw, b3, st = (
            metrics.pairwise(labels, truth),
            metrics.bcubed(labels, truth),
            metrics.cluster_stats(labels, truth),
        )
        r.methods[name] = {
            "pair_precision": pw["precision"],
            "pair_recall": pw["recall"],
            "pair_f1": pw["f1"],
            "bcubed_precision": b3["precision"],
            "bcubed_recall": b3["recall"],
            "bcubed_f1": b3["f1"],
            **st,
        }
        r.lookalikes[name] = {
            k: float(np.mean([labels[i] == labels[j] for i, j in v])) if v else 0.0 for k, v in look.items()
        }
    r.lookalikes["pairs"] = {k: float(len(v)) for k, v in look.items()}
    for thr in (0.3, 0.5, 0.7, 0.9):
        labels = pipeline.average_linkage(n, b.pairs, prob, thr, score)
        pw = metrics.pairwise(labels, truth)
        r.sensitivity.append({"threshold": thr, **pw})
    imp = _permutation_importance(matcher, x, y)
    r.importance = sorted(zip(FEATURES, imp, strict=True), key=lambda t: -t[1])[:8]
    r.seconds = time.perf_counter() - t0
    return r


def _permutation_importance(m: pipeline.Matcher, x: np.ndarray, y: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), min(len(x), 30000), replace=False)
    xs, ys = x[idx], y[idx]
    base = average_precision_score(ys, m.prob(xs))
    out = []
    for k in range(xs.shape[1]):
        xp = xs.copy()
        xp[:, k] = rng.permutation(xp[:, k])
        out.append(float(base - average_precision_score(ys, m.prob(xp))))
    return out


def check(r: Result) -> list[str]:
    p = []
    if r.blocking["pair_completeness"] < 0.98:
        p.append(f"blocking loses too many true pairs: {r.blocking['pair_completeness']:.3f}")
    if r.blocking["reduction_ratio"] < 0.99:
        p.append("blocking does not reduce the comparisons enough")
    m = r.methods
    best = m["matcher + average linkage"]
    for base in ("exact match (email, or name + birth date)", "hand-written rules"):
        if 1 - best["pair_f1"] > 0.5 * (1 - m[base]["pair_f1"]):
            p.append(f"the pipeline should make at most half the pairwise errors of {base}")
    if best["merged_clusters"] >= m["matcher + connected components"]["merged_clusters"]:
        p.append("average linkage should merge fewer different people than connected components")
    if best["pair_f1"] < 0.95:
        p.append(f"pairwise F1 {best['pair_f1']:.3f} < 0.95")
    if r.lookalikes["matcher + average linkage"]["twin"] > 0.1:
        p.append("too many twins merged")
    return p


def to_json(r: Result) -> str:
    return json.dumps(asdict(r), indent=2) + "\n"


def to_markdown(r: Result) -> str:
    bl = r.blocking
    L = [
        "# Evaluation",
        "",
        f"{r.records:,} records of {r.people:,} people from three systems. The matcher is trained on half the people; "
        f"everything below is on the other half ({r.test_records:,} records, {r.test_people:,} people), which it never "
        "saw. Generated by `resolve evaluate`.",
        "",
        "## Blocking",
        "",
        f"{bl['candidate_pairs']:,.0f} candidate pairs out of {bl['all_pairs']:,.0f} possible "
        f"(reduction {bl['reduction_ratio']:.2%}); {bl['pair_completeness']:.2%} of true matching pairs survive. "
        f"{bl['skipped_blocks']:.0f} oversized blocks skipped.",
        "",
        f"Matcher on candidate pairs: average precision {r.matcher['average_precision']:.4f} "
        f"({r.matcher['positives']:.1%} of candidates are true matches).",
        "",
        "## Resolution",
        "",
        "| method | pairwise P | pairwise R | pairwise F1 | B-cubed F1 | clusters (true people) | clusters mixing people | largest |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in r.methods.items():
        L.append(
            f"| {name} | {v['pair_precision']:.3f} | {v['pair_recall']:.3f} | {v['pair_f1']:.3f} | {v['bcubed_f1']:.3f} | "
            f"{v['clusters']:,.0f} ({v['true_entities']:,.0f}) | {v['merged_clusters']:,.0f} | {v['largest_cluster']:.0f} |"
        )
    pairs = r.lookalikes["pairs"]
    L += [
        "",
        "## Look-alikes wrongly merged (share of record pairs)",
        "",
        f"| method | twins ({pairs['twin']:.0f} pairs) | father and son, same name ({pairs['junior']:.0f}) | family at the same address ({pairs['family']:.0f}) |",
        "|---|---:|---:|---:|",
    ]
    for name, v in r.lookalikes.items():
        if name != "pairs":
            L.append(f"| {name} | {v['twin']:.1%} | {v['junior']:.1%} | {v['family']:.1%} |")
    L += [
        "",
        f"## Threshold (average linkage; {THRESHOLD} is used above)",
        "",
        "| threshold | precision | recall | F1 |",
        "|---:|---:|---:|---:|",
    ]
    L += [f"| {s['threshold']:.1f} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |" for s in r.sensitivity]
    L += [
        "",
        "## What the matcher relies on (drop in average precision when a feature is shuffled)",
        "",
        "| feature | importance |",
        "|---|---:|",
    ]
    L += [f"| {f} | {v:.4f} |" for f, v in r.importance]
    L.append("")
    return "\n".join(L)
