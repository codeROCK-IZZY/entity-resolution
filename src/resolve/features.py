"""Normalisation, string similarity and the features that describe a pair of records."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data import NICK_TO_NAME, TYPES, Record

ABBR = {abbr.lower(): full.lower() for full, abbr in TYPES.items()}


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 0.0
    if not a or not b:
        return 0.0
    window = max(0, max(len(a), len(b)) // 2 - 1)
    am = [False] * len(a)
    bm = [False] * len(b)
    matches = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not bm[j] and b[j] == ca:
                am[i] = bm[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t = 0
    k = 0
    for i, ca in enumerate(a):
        if am[i]:
            while not bm[k]:
                k += 1
            if ca != b[k]:
                t += 1
            k += 1
    jaro = (matches / len(a) + matches / len(b) + (matches - t / 2) / matches) / 3
    prefix = 0
    for ca, cb in zip(a[:4], b[:4], strict=False):
        if ca != cb:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def soundex(name: str) -> str:
    name = re.sub(r"[^a-z]", "", name.lower())
    if not name:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    out = name[0].upper()
    prev = codes.get(name[0], "")
    for c in name[1:]:
        code = codes.get(c, "")
        if code and code != prev:
            out += code
        if c not in "hw":
            prev = code
    return (out + "000")[:4]


@dataclass(frozen=True)
class Clean:
    first: str
    first_canon: str  # nickname mapped to the full name
    last: str
    dob: tuple[int, int, int] | None
    email: str  # lowercased, dots and +tags removed from the local part
    phone: str  # last 10 digits
    street: str  # abbreviations expanded
    house: str
    postcode: str


def clean(r: Record) -> Clean:
    first = re.sub(r"[^a-z]", "", r.first.lower())
    last = re.sub(r"[^a-z]", "", r.last.lower())
    dob = tuple(int(x) for x in r.dob.split("-")) if r.dob else None
    email = ""
    if "@" in r.email:
        local, dom = r.email.lower().split("@", 1)
        email = local.split("+")[0].replace(".", "") + "@" + dom
    digits = re.sub(r"\D", "", r.phone)
    phone = ("0" + digits[2:] if digits.startswith("44") else digits)[-10:] if digits else ""
    words = [ABBR.get(w, w) for w in re.findall(r"[a-z0-9]+", r.street.lower())]
    house = words[0] if words and words[0].isdigit() else ""
    street = " ".join(w for w in words if not w.isdigit())
    return Clean(
        first,
        NICK_TO_NAME.get(first, first),
        last,
        dob,  # type: ignore[arg-type]
        email,
        phone,
        street,
        house,
        r.postcode.lower().replace(" ", ""),
    )


FEATURES = [
    "first_jw",
    "first_canon_equal",
    "first_initial_only_match",
    "first_missing",
    "last_jw",
    "last_soundex",
    "names_swapped",
    "dob_equal",
    "dob_day_month_swapped",
    "dob_two_of_three",
    "dob_missing",
    "email_equal",
    "email_local_jw",
    "email_missing",
    "phone_equal",
    "phone_missing",
    "street_jw",
    "house_equal",
    "postcode_equal",
    "address_missing",
    "last_rarity",
    "first_rarity",
]


class PairFeatures:
    """Pair features, including how rare the shared names are (a shared rare surname is strong evidence)."""

    def __init__(self, records: list[Clean]) -> None:
        self.records = records
        n = len(records)
        self.last_freq = Counter(r.last for r in records if r.last)
        self.first_freq = Counter(r.first_canon for r in records if r.first_canon)
        self.n = n

    def _rarity(self, freq: Counter[str], key: str) -> float:
        return -math.log((freq.get(key, 0) + 1) / (self.n + 1))

    def pair(self, i: int, j: int) -> list[float]:
        a, b = self.records[i], self.records[j]
        fj = jaro_winkler(a.first_canon, b.first_canon)
        initial = float(
            bool(a.first and b.first) and (len(a.first) == 1 or len(b.first) == 1) and a.first[0] == b.first[0]
        )
        dob_eq = dob_sw = dob_23 = 0.0
        if a.dob and b.dob:
            dob_eq = float(a.dob == b.dob)
            dob_sw = float(a.dob[0] == b.dob[0] and a.dob[1] == b.dob[2] and a.dob[2] == b.dob[1] and a.dob != b.dob)
            dob_23 = float(sum(x == y for x, y in zip(a.dob, b.dob, strict=True)) == 2)
        el = jaro_winkler(a.email.split("@")[0], b.email.split("@")[0]) if a.email and b.email else 0.0
        same_last = a.last and a.last == b.last
        same_first = a.first_canon and a.first_canon == b.first_canon
        return [
            fj,
            float(bool(same_first)),
            initial,
            float(not a.first or not b.first),
            jaro_winkler(a.last, b.last),
            float(bool(a.last) and soundex(a.last) == soundex(b.last)),
            float(bool(a.first and a.last) and a.first == b.last and a.last == b.first),
            dob_eq,
            dob_sw,
            dob_23,
            float(not a.dob or not b.dob),
            float(bool(a.email) and a.email == b.email),
            el,
            float(not a.email or not b.email),
            float(bool(a.phone) and a.phone == b.phone),
            float(not a.phone or not b.phone),
            jaro_winkler(a.street, b.street),
            float(bool(a.house) and a.house == b.house),
            float(bool(a.postcode) and a.postcode == b.postcode),
            float(not a.street or not b.street),
            self._rarity(self.last_freq, a.last) if same_last else 0.0,
            self._rarity(self.first_freq, a.first_canon) if same_first else 0.0,
        ]

    def matrix(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        return np.array([self.pair(i, j) for i, j in pairs]) if pairs else np.zeros((0, len(FEATURES)))
