from __future__ import annotations

import numpy as np
import pytest

from resolve import data, metrics, pipeline
from resolve.cli import main
from resolve.features import clean, jaro_winkler, soundex


def rec(**kw: str) -> data.Record:
    base = {"first": "", "last": "", "dob": "", "email": "", "phone": "", "street": "", "city": "", "postcode": ""}
    base.update(kw)
    return data.Record(
        0,
        0,
        "crm",
        base["first"],
        base["last"],
        base["dob"],
        base["email"],
        base["phone"],
        base["street"],
        base["city"],
        base["postcode"],
        "",
        0,
    )


def test_string_similarity() -> None:
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.961, abs=1e-3)
    assert jaro_winkler("", "x") == 0.0
    assert soundex("Robert") == soundex("Rupert") == "R163"
    assert soundex("Smith") == soundex("Smyth")


def test_normalisation() -> None:
    c = clean(
        rec(
            first="Bob",
            last="O'Neil",
            email="Bob.Smith+shop@Gmail.com",
            phone="+44 7700 900123",
            street="12 Mill Rd",
            postcode="LS1 4AB",
            dob="1980-03-07",
        )
    )
    assert c.first_canon == "robert"
    assert c.last == "oneil"
    assert c.email == "bobsmith@gmail.com"
    assert c.phone == "07700900123"[-10:]
    assert c.street == "mill road"
    assert c.house == "12"
    assert c.postcode == "ls14ab"
    assert c.dob == (1980, 3, 7)


def test_generator_keeps_truth_and_lookalikes() -> None:
    recs = data.generate(400, seed=1)
    assert len({r.person for r in recs}) == 400
    kinds = {r.kind for r in recs}
    assert {"twin", "junior", "family"} & kinds
    emails = {}
    for r in recs:  # an email address belongs to one person only
        if r.email:
            key = clean(r).email
            assert emails.setdefault(key, r.person) == r.person


def test_blocking_finds_typo_and_swapped_names() -> None:
    a = clean(rec(first="Katherine", last="Fitzgerald", dob="1975-04-02"))
    b = clean(rec(first="Kathryn", last="Fitzgeral", dob="1975-04-02"))
    c = clean(rec(first="Fitzgerald", last="Katherine"))
    pairs = set(pipeline.block([a, b, c]).pairs)
    assert (0, 1) in pairs
    assert (0, 2) in pairs


def test_metrics_on_a_known_clustering() -> None:
    truth = [0, 0, 1, 1]
    assert metrics.pairwise(truth, truth)["f1"] == 1.0
    merged = [0, 0, 0, 0]
    pw = metrics.pairwise(merged, truth)
    assert pw["recall"] == 1.0
    assert pw["precision"] == pytest.approx(2 / 6)
    assert metrics.bcubed([0, 1, 2, 3], truth)["precision"] == 1.0
    assert metrics.cluster_stats(merged, truth)["merged_clusters"] == 1


def test_average_linkage_refuses_a_chain() -> None:
    # 0-1 and 1-2 look like matches, but 0 and 2 clearly are not: connected components chains all three
    pairs = [(0, 1), (1, 2)]
    probs = np.array([0.9, 0.8])
    assert len(set(pipeline.connected_components(3, pairs, probs, 0.5))) == 1
    labels = pipeline.average_linkage(3, pairs, probs, 0.5, lambda i, j: 0.0)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_baselines() -> None:
    cs = [
        clean(rec(first="Ann", last="Lee", email="ann@x.com")),
        clean(rec(first="Anne", last="Lee", email="ann@x.com")),
        clean(rec(first="Ann", last="Lee", dob="1990-01-01")),
        clean(rec(first="Ann", last="Lee", dob="1990-01-01")),
    ]
    labels = pipeline.exact_match_baseline(cs)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_end_to_end_on_a_small_population(tmp_path: object) -> None:
    assert main(["evaluate", "--people", "600", "--out", str(tmp_path)]) == 0


def test_compare_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "compare",
            '{"first": "Bob", "last": "Smith", "dob": "1980-01-02"}',
            '{"first": "Robert", "last": "Smith", "dob": "1980-02-01"}',
        ]
    )
    out = capsys.readouterr().out
    assert "first_canon_equal" in out
    assert "dob_day_month_swapped" in out
