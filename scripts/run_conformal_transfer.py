"""Conformal vs a fixed-k heuristic under dataset shift (local, no GPU).

On a single dataset a calibrated conformal set and a fixed top-k of the same size cover the culprit
equally well, so the guarantee can look like free math on top of a heuristic. We test whether its
value appears under shift: we calibrate a set size on one dataset (or condition) and transfer it
unchanged to another (the fixed-k heuristic), versus recalibrating conformal on the target. On this
data the fixed size does not drift below target, so we report the distribution-free guarantee and the
adaptive set size honestly, not an empirical coverage win. Reuses cached data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from lineup.conformal import build_ranks, calibrate_tau, coverage_and_size
from lineup.data.serialization import read_predictions, read_roles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "conformal_transfer.md"
DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]


def items_for(dataset: str | None = None, condition: str | None = None):
    items = []
    for ds in DATASETS:
        if dataset and ds != dataset:
            continue
        for cond in CONDITIONS:
            if condition and cond != condition:
                continue
            for model in MODELS:
                cell = DATA / ds / cond / model
                if not (cell / "roles.jsonl").exists():
                    continue
                wrong = [c for c in read_roles(cell / "roles.jsonl") if not c.original_correct]
                preds = read_predictions(cell / "predictions.jsonl")
                items += build_ranks(wrong, preds, source=f"{ds}/{cond}/{model}", condition=cond)
    return items


def split(items):
    cal, test = [], []
    for item in items:
        digest = int(hashlib.sha256(item.key.encode()).hexdigest(), 16)
        (cal if digest % 2 == 0 else test).append(item)
    return cal, test


def main() -> None:
    alpha = 0.1
    target = 1 - alpha
    lines = ["# Conformal beats a fixed-k heuristic under dataset shift", ""]
    lines.append(f"Target coverage {target:.2f}. A set size calibrated on the source is transferred unchanged to the")
    lines.append("target (the fixed-k heuristic); conformal instead recalibrates on the target. Same test split.")
    lines.append("")
    lines.append("| shift | source | target | fixed size from source | coverage on target | conformal tau | coverage |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    # dataset shift (mild) and the paper-relevant redundancy shift (baseline -> hard-traps)
    shifts = [
        ("dataset", dict(dataset="hotpotqa"), dict(dataset="2wiki")),
        ("dataset", dict(dataset="2wiki"), dict(dataset="hotpotqa")),
        ("redundancy", dict(condition="baseline"), dict(condition="hardtraps")),
        ("redundancy", dict(condition="hardtraps"), dict(condition="baseline")),
    ]
    for kind, src_f, tgt_f in shifts:
        tau_src = calibrate_tau(items_for(**src_f), alpha)
        tgt_cal, tgt_test = split(items_for(**tgt_f))
        tau_tgt = calibrate_tau(tgt_cal, alpha)
        cov_fixed, _ = coverage_and_size(tgt_test, tau_src)
        cov_conf, _ = coverage_and_size(tgt_test, tau_tgt)
        src_name = "/".join(src_f.values())
        tgt_name = "/".join(tgt_f.values())
        flag = "" if cov_fixed >= target else " (under-covers)"
        lines.append(f"| {kind} | {src_name} | {tgt_name} | {tau_src} | {cov_fixed:.2f}{flag} | {tau_tgt} | {cov_conf:.2f} |")
        print(f"[{kind}] {src_name} -> {tgt_name}: fixed size {tau_src} -> coverage {cov_fixed:.2f} "
              f"(target {target:.2f}); conformal tau {tau_tgt} -> {cov_conf:.2f}")
    lines.append("")
    lines.append("Honest reading: on the single-culprit target a fixed top-2 stays above the 0.90 target across")
    lines.append("every shift here, so conformal does not *empirically* beat the heuristic on this data. We do not")
    lines.append("manufacture a shift that breaks it. Conformal's value is therefore the distribution-free guarantee")
    lines.append("(it holds by construction without knowing the shift) and the self-reporting set size: tau rises")
    lines.append("2 -> 3 as redundancy increases (baseline -> hard-traps), and to the full context to cover an entire")
    lines.append("responsible coalition (run_conformal.py). We claim the guarantee and the adaptive size, not an")
    lines.append("empirical coverage win a fixed top-k cannot match.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
