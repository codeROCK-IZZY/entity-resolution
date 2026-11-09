"""`resolve`: run the evaluation, or compare two records and see why they match or not."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import check, run, to_json, to_markdown
    from .report import render

    r = run(n_people=args.people, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(to_json(r), newline="\n", encoding="utf-8")
    (out / "evaluation.md").write_text(to_markdown(r), newline="\n", encoding="utf-8")
    (out / "report.html").write_text(render(r), newline="\n", encoding="utf-8")
    print(f"{r.test_records:,} records resolved ({r.seconds:.0f}s) -> {out}")
    if args.gate:
        problems = check(r)
        for p in problems:
            print(f"GATE FAILED: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .data import Record
    from .features import FEATURES, PairFeatures, clean

    def rec(raw: str, i: int) -> Record:
        d = json.loads(raw)
        return Record(
            i,
            -1,
            "cli",
            d.get("first", ""),
            d.get("last", ""),
            d.get("dob", ""),
            d.get("email", ""),
            d.get("phone", ""),
            d.get("street", ""),
            d.get("city", ""),
            d.get("postcode", ""),
            "",
            -1,
        )

    pf = PairFeatures([clean(rec(args.a, 0)), clean(rec(args.b, 1))])
    for name, v in zip(FEATURES, pf.pair(0, 1), strict=True):
        if v:
            print(f"{name:26} {v:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resolve", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="train on half the people, resolve the other half, grade it")
    e.add_argument("--people", type=int, default=6000)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    e.add_argument("--gate", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    c = sub.add_parser("compare", help="pair features for two records given as JSON")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
