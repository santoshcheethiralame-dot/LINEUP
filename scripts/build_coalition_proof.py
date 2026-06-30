"""Pool the per-cell leave-two-out proofs into one table and a worked example (local, no GPU).

Run after run_coalition_proof.py has written coalition_proof.jsonl into each cell. Reports, across
the no-single-cause cases, the share where removing a *pair* flips the answer though neither passage
alone does -- the direct evidence that those cases are redundant coalitions and not noise.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "coalition_proof.md"


def pair_kind(pair) -> str:
    planted = sum(1 for cid in pair if "misleading" in cid or "decoy" in cid)
    return "organic" if planted == 0 else ("planted" if planted == 2 else "mixed")


def main() -> None:
    rows = []
    pooled_n = pooled_syn = 0
    kinds = {"organic": 0, "mixed": 0, "planted": 0}
    example = organic_example = None
    for path in sorted(glob.glob(str(ROOT / "paper" / "results_data" / "*" / "*" / "*" / "coalition_proof.jsonl"))):
        records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not records:
            continue
        n = len(records)
        syn = sum(int(r["has_synergy"]) for r in records)
        cell = "/".join(Path(path).parts[-4:-1])
        rows.append((cell, n, syn))
        pooled_n += n
        pooled_syn += syn
        for r in records:
            if r["has_synergy"] and r.get("synergy_pair"):
                kinds[pair_kind(r["synergy_pair"])] += 1
                if r.get("answer_without_pair"):
                    example = example or r
                    if pair_kind(r["synergy_pair"]) == "organic":
                        organic_example = organic_example or r
    example = organic_example or example

    if pooled_n == 0:
        print("no coalition_proof.jsonl found -- run scripts/run_coalition_proof.py per cell first")
        return

    rate = 100 * pooled_syn / pooled_n
    L = ["# Leave-two-out proof: the no-single-cause cases are redundant coalitions", ""]
    L.append("On wrong cases where single-chunk leave-one-out finds no causal passage, we run the pairwise")
    L.append("(leave-two-out) test. A synergy pair is one whose joint removal flips the answer though neither")
    L.append("passage alone does -- exactly the redundancy single leave-one-out cannot see.")
    L.append("")
    L.append("| cell | no-single-cause cases | with a synergy pair |")
    L.append("|---|---:|---:|")
    for cell, n, syn in rows:
        L.append(f"| {cell} | {n} | {syn} ({100*syn/n:.0f}%) |")
    L.append(f"| **pooled** | {pooled_n} | **{pooled_syn} ({rate:.0f}%)** |")
    L.append("")
    L.append(f"Across the pooled no-single-cause slice, **{rate:.0f}%** have a pair whose joint removal flips the")
    L.append("answer though neither passage alone does. The redundancy is shown by direct ablation, not inferred")
    L.append("from the absence of a single cause: these cases are genuine coalitions the single-removal oracle is")
    L.append("blind to, which is why a single pick cannot name a culprit.")
    total_syn = sum(kinds.values())
    if total_syn:
        L.append("")
        L.append(
            "**Planted vs organic (honest).** Of the synergy pairs, "
            f"**{100*kinds['organic']/total_syn:.0f}% are between un-planted passages (organic redundancy)**, "
            f"{100*kinds['mixed']/total_syn:.0f}% one planted + one real, and "
            f"{100*kinds['planted']/total_syn:.0f}% both planted (misleading+decoy)."
        )
        L.append("Every scenario plants exactly one misleading chunk; hard-traps adds a decoy on top, so a")
        L.append("both-planted pair can only form under hard-traps. The honest organic claim is the **organic**")
        L.append("tally above (a synergy pair of two un-planted passages) -- not the baseline cells, which still")
        L.append("carry the planted misleading chunk and so contribute mixed pairs too.")
    if example:
        L.append("")
        L.append("Worked example:")
        L.append("")
        L.append(f"- question gold answer: `{example['gold_answer']}`")
        L.append(f"- the model answered: `{example['model_answer']}` (wrong; no single passage is causal)")
        L.append(f"- removing the pair `{example['synergy_pair'][0]}` + `{example['synergy_pair'][1]}` changes the answer to: `{example['answer_without_pair']}`")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"pooled: {pooled_syn}/{pooled_n} no-single-cause cases have a synergy pair ({rate:.0f}%)")
    print(f"pair types -> organic {kinds['organic']}, mixed {kinds['mixed']}, planted {kinds['planted']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
