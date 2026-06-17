import argparse
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from lineup import __version__
from lineup.config import DEFAULT_MODEL, DEFAULT_SEED, OUTPUT_DIR, PROJECT_ROOT
from lineup.data.serialization import read_roles, read_scenarios
from lineup.release import (
    build_manifest,
    case_records,
    chunk_records,
    data_card,
    dataset_statistics,
)

_PACKAGES = ("transformers", "datasets", "torch", "numpy", "scikit-learn")


def _package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a pipeline run as a release dataset.")
    parser.add_argument("--scenarios", type=Path, default=OUTPUT_DIR / "scenarios.jsonl")
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "release")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-dataset", default="hotpotqa/hotpot_qa")
    parser.add_argument("--source-split", default="validation")
    args = parser.parse_args()

    scenarios = read_scenarios(args.scenarios)
    cases = read_roles(args.roles)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out / "cases.jsonl", case_records(scenarios, cases))
    _write_jsonl(args.out / "chunks.jsonl", chunk_records(scenarios, cases))

    statistics = dataset_statistics(scenarios, cases)
    card_stats = dict(statistics)
    card_stats.update(
        model=args.model,
        seed=args.seed,
        source_dataset=args.source_dataset,
        source_split=args.source_split,
    )
    (args.out / "README.md").write_text(data_card(card_stats), encoding="utf-8")

    manifest = build_manifest(
        lineup_version=__version__,
        model=args.model,
        seed=args.seed,
        source_dataset=args.source_dataset,
        source_split=args.source_split,
        created=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        packages=_package_versions(),
        statistics=statistics,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    roles = statistics["role_counts"]
    print(
        f"packaged {statistics['n_cases']} cases "
        f"({statistics['n_labeled_cases']} labeled, {statistics['n_chunks']} passages)"
    )
    print(f"roles: {roles}")
    for name in ("cases.jsonl", "chunks.jsonl", "README.md", "manifest.json"):
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
