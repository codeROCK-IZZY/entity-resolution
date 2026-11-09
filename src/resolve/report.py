"""A self-contained HTML report of the evaluation."""

from __future__ import annotations

import html

from .evaluate import Result

CSS = """
:root { --bg:#0b0e13; --panel:#121720; --line:#232c3a; --ink:#e8ecf2; --ink2:#9aa6b8; --s1:#3987e5; --s2:#d95926;
  --good:#3fcf8e; --bad:#ff6b6b; --mid:#ffb454; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }
h1 { font-size: 30px; letter-spacing:-0.02em; margin: 0 0 6px; } h2 { font-size: 19px; margin: 34px 0 14px; }
.sub, .muted { color: var(--ink2); } .sub { margin: 0 0 24px; }
.tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.tile b { display:block; font-size: 28px; letter-spacing:-0.02em; } .tile span { color: var(--ink2); font-size: 13px; }
table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size: 13.5px; }
th, td { padding: 8px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; } th { color: var(--ink2); font-weight:500; font-size:12px; }
td.good { color: var(--good); font-weight:600; } td.bad { color: var(--bad); font-weight:600; } td.mid { color: var(--mid); font-weight:600; }
tr.best td { background: rgba(57,135,229,.10); }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }
"""


def _merge_cls(x: float) -> str:
    return "good" if x <= 0.02 else ("mid" if x <= 0.1 else "bad")


def render(r: Result) -> str:
    best = r.methods["matcher + average linkage"]
    rules = r.methods["hand-written rules"]
    cc = r.methods["matcher + connected components"]
    tiles = [
        (f"{best['pair_f1']:.3f}", f"pairwise F1 on people the matcher never saw (rules: {rules['pair_f1']:.3f})"),
        (f"{(1 - rules['pair_f1']) / (1 - best['pair_f1']):.1f}x", "fewer pairwise errors than hand-written rules"),
        (
            f"{best['merged_clusters']:.0f}",
            f"clusters mixing different people (connected components: {cc['merged_clusters']:.0f})",
        ),
        (
            f"{r.blocking['pair_completeness']:.1%}",
            f"of true pairs kept while skipping {r.blocking['reduction_ratio']:.2%} of comparisons",
        ),
    ]
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    rows = "".join(
        f'<tr class="{"best" if n == "matcher + average linkage" else ""}"><td>{html.escape(n)}</td><td>{v["pair_precision"]:.3f}</td>'
        f"<td>{v['pair_recall']:.3f}</td><td>{v['pair_f1']:.3f}</td><td>{v['bcubed_f1']:.3f}</td>"
        f"<td>{v['clusters']:,.0f}</td><td>{v['merged_clusters']:,.0f}</td></tr>"
        for n, v in r.methods.items()
    )
    look = "".join(
        f"<tr><td>{html.escape(n)}</td>"
        + "".join(f'<td class="{_merge_cls(v[k])}">{v[k]:.1%}</td>' for k in ("twin", "junior", "family"))
        + "</tr>"
        for n, v in r.lookalikes.items()
        if n != "pairs"
    )
    sens = "".join(
        f"<tr><td>{s['threshold']:.1f}</td><td>{s['precision']:.3f}</td><td>{s['recall']:.3f}</td><td>{s['f1']:.3f}</td></tr>"
        for s in r.sensitivity
    )
    imp = "".join(f"<tr><td>{html.escape(f)}</td><td>{v:.4f}</td></tr>" for f, v in r.importance)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Entity Resolution: evaluation</title><style>{CSS}</style></head><body><main>
<h1>Who is who, measured</h1>
<p class="sub">{r.records:,} customer records of {r.people:,} people from three systems. The matcher learns from half the
people; everything here is on the other {r.test_people:,}, with twins, fathers and sons, and families at one address.</p>
<div class="tiles">{t}</div>
<h2>Resolution ({r.test_records:,} records, {r.test_people:,} true people)</h2>
<table><tr><th>method</th><th>pair precision</th><th>pair recall</th><th>pair F1</th><th>B-cubed F1</th><th>clusters</th>
<th>clusters mixing people</th></tr>{rows}</table>
<h2>Look-alikes wrongly merged</h2>
<table><tr><th>method</th><th>twins</th><th>father and son, same name</th><th>family at the same address</th></tr>{look}</table>
<div class="cols"><div><h2>Threshold (average linkage)</h2>
<table><tr><th>threshold</th><th>precision</th><th>recall</th><th>F1</th></tr>{sens}</table></div>
<div><h2>What the matcher relies on</h2><table><tr><th>feature</th><th>drop in average precision when shuffled</th></tr>{imp}</table></div></div>
</main></body></html>
"""
