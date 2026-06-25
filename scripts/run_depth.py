"""Retrieval-depth sweep: does ill-posedness grow as you retrieve more passages? (natural slice, qwen)

Finding: retrieving more passages makes the model err more (the error rate climbs), but the FRACTION
of those errors that are ill-posed (no single culprit) is stable across k=4..10. So no-culprit is not
an artifact of retrieval depth. Appendix figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures" / "fig14_depth.png"
OUT = ROOT / "paper" / "depth.md"
# k -> roles dir (k=6 is the already-ingested natural slice)
POINTS = {
    4: ROOT / "paper" / "depth_data" / "k4" / "qwen",
    6: ROOT / "paper" / "natural_data" / "hotpotqa" / "qwen",
    8: ROOT / "paper" / "depth_data" / "k8" / "qwen",
    10: ROOT / "paper" / "depth_data" / "k10" / "qwen",
}


def stats(d):
    rows = [json.loads(l) for l in (d / "roles.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)
    wrong = [r for r in rows if not r["original_correct"]]
    nc = sum(1 for r in wrong if not any(c["causal"] and c["salient"] for c in r["chunk_roles"]))
    return total, len(wrong), (nc / len(wrong) if wrong else 0.0)


def main():
    ks, err, ncr, rows = [], [], [], []
    for k in sorted(POINTS):
        total, nw, nc = stats(POINTS[k])
        ks.append(k); err.append(100 * nw / total); ncr.append(100 * nc); rows.append((k, total, nw, 100 * nw / total, 100 * nc))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, err, "-o", color="#c0653a", label="error rate (% of cases wrong)")
    ax.plot(ks, ncr, "-s", color="#3b6fb6", label="no-culprit rate (% of errors ill-posed)")
    ax.set_xticks(ks)
    ax.set_xlabel("retrieval depth k (passages in context)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 45)
    ax.set_title("More retrieval -> more errors, but a steady ill-posed fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)

    L = ["# Retrieval-depth sweep (natural slice, HotpotQA/qwen)", ""]
    L.append("| k | cases | wrong | error rate % | no-culprit % |")
    L.append("|---:|---:|---:|---:|---:|")
    for k, total, nw, e, nc in rows:
        L.append(f"| {k} | {total} | {nw} | {e:.0f} | {nc:.0f} |")
    L.append("")
    lo, hi = min(ncr), max(ncr)
    L.append(f"As retrieval depth grows from 4 to 10, the **error rate climbs** ({err[0]:.0f}% -> {err[-1]:.0f}%)")
    L.append(f"-- more passages give the model more ways to go wrong -- but the **no-culprit fraction is flat**")
    L.append(f"([{lo:.0f}%, {hi:.0f}%]). Ill-posedness is therefore not an artifact of how much you retrieve;")
    L.append("a steady ~quarter of errors have no single culprit regardless of context size. (Fig 14, appendix.)")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print("depth sweep (natural, hotpotqa/qwen):")
    for k, total, nw, e, nc in rows:
        print(f"  k={k:2d}  cases {total}  wrong {nw}  error {e:.0f}%  no-culprit {nc:.0f}%")
    print(f"wrote {FIG}\nwrote {OUT}")


if __name__ == "__main__":
    main()
