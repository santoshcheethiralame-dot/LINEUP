"""#1 Does ill-posedness track reasoning structure? (local)

Each scenario carries meta.type (bridge / comparison / ...) and meta.level. We test whether the
no-culprit rate splits by question type -- i.e. whether ill-posedness is a fingerprint of the
reasoning structure (comparison questions need two facts -> two causes -> structurally more
no-culprit) rather than noise. Reported per dataset (the type taxonomies differ) and also for the
baseline condition alone, to show any effect is not an artifact of hard-traps planting.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "structure.md"
FIG = ROOT / "paper" / "figures" / "fig11_structure.png"


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (center - half), 100 * (center + half))

MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]


def no_culprit(case):
    return not any(c["causal"] and c["salient"] for c in case["chunk_roles"])


def collect(datasets, conds):
    # (dataset, type) -> [n_wrong, n_no_culprit]
    agg = defaultdict(lambda: [0, 0])
    for ds in datasets:
        for cond in conds:
            for model in MODELS:
                d = DATA / ds / cond / model
                rp, sp = d / "roles.jsonl", d / "scenarios.jsonl"
                if not (rp.exists() and sp.exists()):
                    continue
                qtype = {}
                for l in sp.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        s = json.loads(l)
                        qtype[s["qid"]] = (s.get("meta") or {}).get("type", "unknown")
                for l in rp.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        r = json.loads(l)
                        if r.get("original_correct", True):
                            continue
                        t = qtype.get(r["qid"], "unknown")
                        agg[(ds, t)][0] += 1
                        agg[(ds, t)][1] += int(no_culprit(r))
    return agg


def section(title, agg):
    lines = [f"### {title}", "", "| dataset | question type | no-culprit % | 95% CI | n |", "|---|---|---:|---:|---:|"]
    for ds in DATASETS:
        rows = sorted([(t, v) for (d, t), v in agg.items() if d == ds], key=lambda x: -x[1][0])
        for t, (nw, nc) in rows:
            if nw >= 20:
                lo, hi = wilson(nc, nw)
                lines.append(f"| {ds} | {t} | {100*nc/nw:.1f} | [{lo:.0f}, {hi:.0f}] | {nw} |")
    lines.append("")
    return lines


def make_figure(full):
    # group by dataset, order types by no-culprit rate; show Wilson CIs
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, ds in zip(axes, DATASETS):
        rows = [(t, nc, nw) for (d, t), (nw, nc) in full.items() if d == ds and nw >= 20]
        rows.sort(key=lambda r: r[1] / r[2])
        labels = [t for t, _, _ in rows]
        rates = [100 * nc / nw for _, nc, nw in rows]
        errs = [[100 * nc / nw - wilson(nc, nw)[0] for _, nc, nw in rows],
                [wilson(nc, nw)[1] - 100 * nc / nw for _, nc, nw in rows]]
        colors = ["#3b6fb6" if ("compar" in t) else "#b8b8b8" for t in labels]
        ax.bar(range(len(rows)), rates, yerr=errs, color=colors, capsize=4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{t}\n(n={nw})" for (t, _, nw) in rows], fontsize=8)
        ax.set_title(ds)
        ax.set_ylim(0, 75)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("errors with no single culprit (%)")
    fig.suptitle("Ill-posedness tracks reasoning structure (comparison-type in blue)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)


def main():
    full = collect(DATASETS, CONDS)
    base = collect(DATASETS, ["baseline"])
    make_figure(full)

    L = ["# Does ill-posedness track reasoning structure?", ""]
    L += section("All wrong cases (pooled over models + conditions)", full)
    L += section("Baseline only (no hard-traps planting)", base)
    L.append("Reading: comparison-style questions (hotpotqa *comparison*, 2wiki *bridge_comparison*) show a")
    L.append("markedly higher no-culprit rate than single-chain ones (2wiki *inference*, hotpotqa *bridge*).")
    L.append("Comparison questions need facts about two entities -> two necessary causes -> no single culprit:")
    L.append("ill-posedness is partly a fingerprint of multi-entity reasoning, not just our construction. The")
    L.append("baseline-only panel rules out a hard-traps artifact. Types with n<20 omitted. (Fig 11.)")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print("=== no-culprit by question type (pooled all cells) ===")
    for ds in DATASETS:
        rows = sorted([(t, v) for (d, t), v in full.items() if d == ds], key=lambda x: -x[1][0])
        for t, (nw, nc) in rows:
            if nw >= 20:
                print(f"  {ds:9s} {t:22s} no-culprit {100*nc/nw:5.1f}%  (n={nw})")
    print("=== baseline only ===")
    for ds in DATASETS:
        rows = sorted([(t, v) for (d, t), v in base.items() if d == ds], key=lambda x: -x[1][0])
        for t, (nw, nc) in rows:
            if nw >= 20:
                print(f"  {ds:9s} {t:22s} no-culprit {100*nc/nw:5.1f}%  (n={nw})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
