"""Redundancy dose-response: as the number of redundant decoys grows (0,1,2,3), single-chunk
attribution degrades while the effect-set holds, and the no-single-culprit rate rises.

Ingest the sweep zips (named lineup_<dataset>_<baseline|hardtraps|dose2|dose3>_<model>.zip), then
this plots recall@1, recall@k, and the no-culprit rate against the decoy count.
"""
import argparse
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lineup.data.serialization import read_predictions, read_roles
from lineup.setvalued import attribution_recovery

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "dose_data"
FIG = ROOT / "paper" / "figures"


def cond_to_n(cond):
    if cond == "baseline":
        return 0
    if cond == "hardtraps":
        return 1
    match = re.match(r"dose(\d+)", cond)
    return int(match.group(1)) if match else None


def ingest(zips):
    for path in zips:
        match = re.match(r"lineup_(hotpotqa|2wiki)_(baseline|hardtraps|dose\d+)_", Path(path).stem)
        if not match:
            print(f"  skip (name): {path}")
            continue
        n = cond_to_n(match.group(2))
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            for model in sorted({m.split("/")[0] for m in members if "/" in m}):
                dest = DATA / str(n) / model
                dest.mkdir(parents=True, exist_ok=True)
                for stem in ("roles", "predictions", "scenarios", "generations"):
                    member = f"{model}/{stem}.jsonl"
                    if member in members:
                        dest.joinpath(f"{stem}.jsonl").write_bytes(archive.read(member))
                print(f"  ingested n={n}  {model}")


def report(model):
    ns, no_culprit, recall_1, recall_k = [], [], [], []
    for n in sorted(int(p.name) for p in DATA.iterdir() if p.is_dir()):
        cell = DATA / str(n) / model
        if not (cell / "roles.jsonl").exists():
            continue
        wrong = [c for c in read_roles(cell / "roles.jsonl") if not c.original_correct]
        preds = read_predictions(cell / "predictions.jsonl")
        rate = sum(1 for c in wrong if not any(r.role == "culprit" for r in c.chunk_roles)) / len(wrong)
        cc = {r.method: r for r in attribution_recovery(wrong, preds)}.get("contextcite")
        ns.append(n)
        no_culprit.append(rate)
        recall_1.append(cc.recall_at_1)
        recall_k.append(cc.recall_at_k)
        print(f"  n={n}: no-culprit {rate:.0%}  recall@1 {cc.recall_at_1:.2f}  recall@k {cc.recall_at_k:.2f}")

    if not ns:
        raise SystemExit("no dose cells found — ingest the sweep zips first")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, recall_1, marker="o", color="#c2c6cc", label="recall@1 (single pick)")
    ax.plot(ns, recall_k, marker="o", color="#3b6fb6", label="recall@k (effect set)")
    ax.plot(ns, no_culprit, marker="s", ls="--", color="#e08a1e", label="no-culprit rate")
    ax.set_xlabel("number of redundant decoys")
    ax.set_ylabel("rate")
    ax.set_xticks(ns)
    ax.set_ylim(0, 1)
    ax.set_title("As redundancy grows, a single pick decays; the set holds")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_doseresponse.png", dpi=150)
    print(f"\nwrote {FIG / 'fig7_doseresponse.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", nargs="+", default=None)
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    if args.ingest:
        ingest(args.ingest)
    report(args.model)


if __name__ == "__main__":
    main()
