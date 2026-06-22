"""Natural-context (real-retriever) slice: do organic errors — with the dataset's own distractors
and no planted near-miss — show the same ill-posedness as the constructed cases?

Ingest the natural run (lineup_<dataset>_natural_<model>.zip), then print error rate, no-culprit
rate, and ContextCite top-1 accuracy side by side with the constructed baseline.
"""
import argparse
import re
import zipfile
from pathlib import Path

from lineup.data.serialization import read_predictions, read_roles
from lineup.scoring import score_predictions

ROOT = Path(__file__).resolve().parent.parent
NAT = ROOT / "paper" / "natural_data"
MATRIX = ROOT / "paper" / "results_data"


def ingest(zips):
    seen = []
    for path in zips:
        match = re.match(r"lineup_(hotpotqa|2wiki)_natural_(.+)", Path(path).stem)
        if not match:
            print(f"  skip (name): {path}")
            continue
        dataset = match.group(1)
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            for model in sorted({m.split("/")[0] for m in members if "/" in m}):
                dest = NAT / dataset / model
                dest.mkdir(parents=True, exist_ok=True)
                for stem in ("roles", "predictions", "scenarios", "generations"):
                    member = f"{model}/{stem}.jsonl"
                    if member in members:
                        dest.joinpath(f"{stem}.jsonl").write_bytes(archive.read(member))
                print(f"  ingested natural {dataset}/{model}")
                seen.append((dataset, model))
    return seen


def _summary(cell):
    roles = read_roles(cell / "roles.jsonl")
    total = len(roles)
    wrong = [c for c in roles if not c.original_correct]
    if not wrong:
        return total, 0, None, None
    no_culprit = sum(1 for c in wrong if not any(r.role == "culprit" for r in c.chunk_roles)) / len(wrong)
    preds = read_predictions(cell / "predictions.jsonl")
    cc = next((r for r in score_predictions(wrong, preds) if r.method == "contextcite"), None)
    return total, len(wrong), no_culprit, (cc.top1_culprit_accuracy if cc else None)


def report(dataset, model):
    print(f"\n{dataset} / {model} — organic vs constructed:")
    print(f"  {'slice':24s} {'errors':>14s}  {'no-culprit':>10s}  {'cc top-1':>9s}")
    for label, cell in [("natural (no planting)", NAT / dataset / model), ("constructed baseline", MATRIX / dataset / "baseline" / model)]:
        if not (cell / "roles.jsonl").exists():
            print(f"  {label:24s} {'(not present)':>14s}")
            continue
        total, wrong, no_culprit, top1 = _summary(cell)
        errors = f"{wrong}/{total} ({wrong / total:.0%})" if total else "n/a"
        nc = f"{no_culprit:.0%}" if no_culprit is not None else "n/a"
        t1 = f"{top1:.2f}" if top1 is not None else "n/a"
        print(f"  {label:24s} {errors:>14s}  {nc:>10s}  {t1:>9s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", nargs="+", default=None)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    targets = ingest(args.ingest) if args.ingest else [(args.dataset, args.model)]
    for dataset, model in targets:
        report(dataset, model)


if __name__ == "__main__":
    main()
