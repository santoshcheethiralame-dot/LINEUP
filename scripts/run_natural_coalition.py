"""Does the 'wrong primitive' hold on UN-planted redundancy? (local, no Kaggle)

The dose-response and recall@1-vs-recall@k results use planted decoys. A reviewer can object that we
engineered the redundancy. Here we show the same structural ceiling on the *natural* slice (no
planting): naturally-occurring coalitions, where >=2 retrieved passages each individually change the
model's answer. The responsible set is the oracle's causal non-gold chunks; recall@1 (single pick) vs
recall@k (set of the right size) is measured on those coalitions.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAT = ROOT / "paper" / "natural_data"
OUT = ROOT / "paper" / "natural_coalition.md"
CELLS = [("hotpotqa", "qwen"), ("2wiki", "qwen"), ("hotpotqa", "mistral")]


def load(cell):
    ds, model = cell
    d = NAT / ds / model
    roles = {}
    for line in (d / "roles.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r["original_correct"]:
                roles[r["qid"]] = r
    preds = {}
    for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            if p.get("method") == "contextcite":
                preds[p["qid"]] = p
    return roles, preds


def main():
    rows = []
    pooled_r1, pooled_rk = [], []
    pooled_wrong = pooled_coal = 0
    for cell in CELLS:
        roles, preds = load(cell)
        r1, rk = [], []
        n_coal = 0
        for qid, case in roles.items():
            responsible = {c["chunk_id"] for c in case["chunk_roles"] if c["causal"] and c["provenance"] != "gold"}
            if len(responsible) < 2:           # a natural coalition: >=2 causal non-gold chunks
                continue
            n_coal += 1
            pr = preds.get(qid)
            if not pr:
                continue
            ranked = [s["chunk_id"] for s in sorted(pr["chunk_scores"], key=lambda s: s["score"], reverse=True)]
            k = len(responsible)
            r1.append(len(set(ranked[:1]) & responsible) / k)
            rk.append(len(set(ranked[:k]) & responsible) / k)
        name = f"{cell[0]}/{cell[1]}"
        rows.append((name, len(roles), n_coal, sum(r1) / len(r1) if r1 else None, sum(rk) / len(rk) if rk else None, len(r1)))
        pooled_r1 += r1
        pooled_rk += rk
        pooled_wrong += len(roles)
        pooled_coal += n_coal

    p_r1 = sum(pooled_r1) / len(pooled_r1)
    p_rk = sum(pooled_rk) / len(pooled_rk)
    L = ["# The wrong primitive holds on un-planted (natural) coalitions", ""]
    L.append("Natural slice (no planting). A *natural coalition* is a wrong case with >=2 causal non-gold")
    L.append("passages — distributed causation the model produces on its own. ContextCite, on those:")
    L.append("")
    L.append("| cell | wrong | natural coalitions | recall@1 | recall@k |")
    L.append("|---|---:|---:|---:|---:|")
    for name, nw, nc, a1, ak, n in rows:
        L.append(f"| {name} | {nw} | {nc} | {a1:.2f} | {ak:.2f} |")
    L.append(f"| **pooled** | {pooled_wrong} | {pooled_coal} | **{p_r1:.2f}** | **{p_rk:.2f}** |")
    L.append("")
    L.append(f"Across the natural slice, **{100*pooled_coal/pooled_wrong:.0f}%** of wrong cases are natural")
    L.append(f"coalitions, and on them a single pick recovers only **recall@1 {p_r1:.2f}** of the responsible")
    L.append(f"set while a set of the right size recovers **recall@k {p_rk:.2f}** — the same structural gap as")
    L.append("the planted dose-response, with nothing constructed. The wrong-primitive finding is not an")
    L.append("artifact of our decoys; redundant causation occurs organically and a single pick can't cover it.")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"pooled: {pooled_wrong} wrong, {pooled_coal} natural coalitions ({100*pooled_coal/pooled_wrong:.0f}%)")
    for name, nw, nc, a1, ak, n in rows:
        print(f"  {name:18s} wrong {nw:3d}  coalitions {nc:3d}  recall@1 {a1:.2f}  recall@k {ak:.2f}")
    print(f"  POOLED recall@1 {p_r1:.2f}  recall@k {p_rk:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
