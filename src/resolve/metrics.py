"""Pairwise and B-cubed precision, recall and F1 for a clustering against the truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations


def true_pairs(truth: list[int]) -> set[tuple[int, int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(truth):
        groups[t].append(i)
    return {p for g in groups.values() for p in combinations(sorted(g), 2)}


def pairwise(pred: list[int], truth: list[int]) -> dict[str, float]:
    tp_set = true_pairs(truth)
    pred_set = true_pairs(pred)
    tp = len(tp_set & pred_set)
    p = tp / len(pred_set) if pred_set else 1.0
    r = tp / len(tp_set) if tp_set else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def bcubed(pred: list[int], truth: list[int]) -> dict[str, float]:
    by_pred: dict[int, list[int]] = defaultdict(list)
    by_true: dict[int, list[int]] = defaultdict(list)
    for i, (c, t) in enumerate(zip(pred, truth, strict=True)):
        by_pred[c].append(i)
        by_true[t].append(i)
    joint = Counter(zip(pred, truth, strict=True))
    p = sum(joint[(c, t)] / len(by_pred[c]) for c, t in zip(pred, truth, strict=True)) / len(pred)
    r = sum(joint[(c, t)] / len(by_true[t]) for c, t in zip(pred, truth, strict=True)) / len(pred)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}


def cluster_stats(pred: list[int], truth: list[int]) -> dict[str, float]:
    people_in: dict[int, set[int]] = defaultdict(set)
    for c, t in zip(pred, truth, strict=True):
        people_in[c].add(t)
    sizes = Counter(pred)
    return {
        "clusters": len(sizes),
        "true_entities": len(set(truth)),
        "merged_clusters": sum(len(v) > 1 for v in people_in.values()),
        "largest_cluster": max(sizes.values()),
    }
