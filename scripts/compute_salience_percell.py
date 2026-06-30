"""Per-cell no-culprit rate under shipped/strict/loose salience, with Wilson 95% CIs (local).

Extends run_salience_sensitivity.py to report each cell (not just pooled), so we can show the
no-single-culprit finding survives in every cell even under the over-generous LOOSE matcher, and
attach a Wilson interval to each rate. Addresses reviewer items #4 (loose-matcher survival) and #13
(Wilson CIs).
"""
from __future__ import annotations
import json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "salience_percell.md"
_WORD = re.compile(r"\w+")
def tokens(s): return {w for w in _WORD.findall(s.lower()) if len(w) >= 4}

def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

def cell_counts(d):
    text = {s["qid"]: {c["chunk_id"]: c["text"] for c in s["chunks"]}
            for s in map(json.loads, filter(str.strip, (d/"scenarios.jsonl").read_text(encoding="utf-8").splitlines()))}
    ans = {g["qid"]: g.get("model_answer","") for g in
           map(json.loads, filter(str.strip, (d/"generations.jsonl").read_text(encoding="utf-8").splitlines()))}
    n = 0; nc = {"shipped":0,"strict":0,"loose":0}
    for r in map(json.loads, filter(str.strip, (d/"roles.jsonl").read_text(encoding="utf-8").splitlines())):
        if r["original_correct"]: continue
        n += 1
        a = (ans.get(r["qid"]) or "").strip(); atok = tokens(a); ct = text.get(r["qid"], {})
        has = {"shipped":False,"strict":False,"loose":False}
        for c in r["chunk_roles"]:
            if not c["causal"]: continue
            body = ct.get(c["chunk_id"], "").lower()
            if c["salient"]: has["shipped"] = True
            if a and a.lower() in body: has["strict"] = True
            if atok and (atok & tokens(body)): has["loose"] = True
        for k in nc:
            if not has[k]: nc[k] += 1
    return n, nc

rows = []; tot_n = 0; tot = {"shipped":0,"strict":0,"loose":0}
for ds in ["hotpotqa","2wiki"]:
    for cond in ["baseline","hardtraps"]:
        for model in ["qwen","phi","mistral"]:
            d = DATA/ds/cond/model
            if not all((d/f).exists() for f in ("roles.jsonl","scenarios.jsonl","generations.jsonl")): continue
            n, nc = cell_counts(d)
            rows.append((f"{ds}/{cond}/{model}", n, nc))
            tot_n += n
            for k in tot: tot[k] += nc[k]

L = ["# Per-cell no-culprit rate by salience matcher (Wilson 95% CIs)", "",
     "No single culprit = no causal passage that is also salient. STRICT = answer verbatim; shipped =",
     "phrase match (reported headline); LOOSE = any shared 4+ char token (over-generous floor).", "",
     "| cell | n | shipped % [95% CI] | loose % (floor) [95% CI] |", "|---|--:|--:|--:|"]
for cell, n, nc in rows:
    sl, sh = wilson(nc["shipped"], n); ll, lh = wilson(nc["loose"], n)
    L.append(f"| {cell} | {n} | {100*nc['shipped']/n:.0f} [{sl:.0f}–{sh:.0f}] | {100*nc['loose']/n:.0f} [{ll:.0f}–{lh:.0f}] |")
sl, sh = wilson(tot["shipped"], tot_n); stl, sth = wilson(tot["strict"], tot_n); ll, lh = wilson(tot["loose"], tot_n)
L.append(f"| **pooled** | {tot_n} | **{100*tot['shipped']/tot_n:.0f} [{sl:.0f}–{sh:.0f}]** | **{100*tot['loose']/tot_n:.0f} [{ll:.0f}–{lh:.0f}]** |")
L += ["", f"Pooled band: strict {100*tot['strict']/tot_n:.0f}%, shipped {100*tot['shipped']/tot_n:.0f}%, loose {100*tot['loose']/tot_n:.0f}%.",
      "**Survival:** every cell's LOOSE-matcher lower CI is the strongest test — the floor under the most",
      "over-generous salience rule. If those stay clear of 0, the no-single-culprit finding survives the",
      "harshest reading of the soft axis."]
OUT.write_text("\n".join(L)+"\n", encoding="utf-8")
print("\n".join(L))
print(f"\nwrote {OUT}")
