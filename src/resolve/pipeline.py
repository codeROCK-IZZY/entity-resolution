"""Blocking, matching and clustering, plus the two baselines every pipeline should beat.

blocking   records share at least one key: (surname sound, first initial), email, phone,
           (birth date, surname initial), (postcode, house number), (birth year, first name);
           blocks larger than a cap are skipped and counted, so a common key can't explode the pairs
matcher    gradient-boosted classifier on pair features, trained on people disjoint from the test set
clustering average-linkage agglomeration: two clusters merge only if their records match *on average*,
           so one wrong link cannot chain two people together (connected components can)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import Clean, PairFeatures, jaro_winkler, soundex

BLOCK_CAP = 150


def block_keys(c: Clean) -> list[str]:
    keys = []
    if c.last and c.first:
        keys.append(f"n:{soundex(c.last)}:{c.first[0]}")
        keys.append(f"n:{soundex(c.first)}:{c.last[0]}")  # first and last swapped
    if c.email:
        keys.append(f"e:{c.email}")
    if c.phone:
        keys.append(f"p:{c.phone}")
    if c.dob and c.last:
        keys.append(f"d:{c.dob}:{c.last[0]}")
    if c.dob and c.first_canon:
        keys.append(f"y:{c.dob[0]}:{c.first_canon}")
    if c.postcode and c.house:
        keys.append(f"a:{c.postcode}:{c.house}")
    return keys


@dataclass
class Blocking:
    pairs: list[tuple[int, int]]
    skipped_blocks: int
    largest_block: int


def block(records: list[Clean], cap: int = BLOCK_CAP) -> Blocking:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(records):
        for k in block_keys(c):
            groups[k].append(i)
    pairs: set[tuple[int, int]] = set()
    skipped = 0
    largest = 0
    for members in groups.values():
        if len(members) > cap:
            skipped += 1
            continue
        largest = max(largest, len(members))
        pairs.update(combinations(sorted(members), 2))
    return Blocking(sorted(pairs), skipped, largest)


@dataclass
class Matcher:
    seed: int = 0
    model: HistGradientBoostingClassifier = field(
        default_factory=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0
        )
    )

    def fit(self, x: np.ndarray, y: np.ndarray) -> Matcher:
        self.model.fit(x, y)
        return self

    def prob(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.model.predict_proba(x)[:, 1] if len(x) else np.zeros(0)
        return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def labels(self) -> list[int]:
        return [self.find(i) for i in range(len(self.parent))]


def connected_components(n: int, pairs: list[tuple[int, int]], probs: np.ndarray, threshold: float) -> list[int]:
    uf = UnionFind(n)
    for (i, j), p in zip(pairs, probs, strict=True):
        if p >= threshold:
            uf.union(i, j)
    return uf.labels()


def average_linkage(
    n: int, pairs: list[tuple[int, int]], probs: np.ndarray, threshold: float, score: Callable[[int, int], float]
) -> list[int]:
    """Greedy agglomeration: take candidate links strongest first; merge two clusters only if the mean match
    probability over *all* record pairs between them clears the threshold (unscored pairs are scored on demand)."""
    cluster = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    known = {(min(i, j), max(i, j)): float(p) for (i, j), p in zip(pairs, probs, strict=True)}

    def p_of(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in known:
            known[key] = score(*key)
        return known[key]

    for idx in np.argsort(-probs):
        if probs[idx] < threshold:
            break
        a, b = cluster[pairs[idx][0]], cluster[pairs[idx][1]]
        if a == b:
            continue
        ma, mb = members[a], members[b]
        mean = float(np.mean([p_of(i, j) for i in ma for j in mb]))
        if mean >= threshold:
            if len(ma) < len(mb):
                a, b, ma, mb = b, a, mb, ma
            for i in mb:
                cluster[i] = a
            ma.extend(mb)
            del members[b]
    return cluster


def exact_match_baseline(records: list[Clean]) -> list[int]:
    """Link records with the same email, or the same first name, surname and birth date."""
    uf = UnionFind(len(records))
    seen: dict[str, int] = {}
    for i, c in enumerate(records):
        keys = []
        if c.email:
            keys.append("e:" + c.email)
        if c.first and c.last and c.dob:
            keys.append(f"n:{c.first}|{c.last}|{c.dob}")
        for k in keys:
            if k in seen:
                uf.union(i, seen[k])
            else:
                seen[k] = i
    return uf.labels()


def rules_baseline(records: list[Clean], pairs: list[tuple[int, int]]) -> list[int]:
    """Hand-written rules a team might start with, over the same candidate pairs."""
    uf = UnionFind(len(records))
    for i, j in pairs:
        a, b = records[i], records[j]
        names = jaro_winkler(a.first_canon, b.first_canon) > 0.9 and jaro_winkler(a.last, b.last) > 0.9
        if (
            (a.email and a.email == b.email)
            or (names and a.dob and a.dob == b.dob)
            or (names and a.phone and a.phone == b.phone)
            or (names and a.postcode and a.postcode == b.postcode)
        ):
            uf.union(i, j)
    return uf.labels()


def pair_features(records: list[Clean]) -> PairFeatures:
    return PairFeatures(records)
