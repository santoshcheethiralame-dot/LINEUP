"""Honest floor for the N=30 confirmatory validation round, with blind adjudication (local, no GPU).

Human ground truth per case: where both reviewers agree, their shared pick; where they clashed, the
blind-adjudicated `correct_answer` (decided WITHOUT seeing the oracle's pick). The oracle is counted
correct only where it matches that human truth. Ambiguous or unfilled adjudications count *against*
the oracle — the honest floor. Cases where both reviewers independently agree but differ from the
oracle are oracle errors that need no adjudication.

    python scripts/score_round30.py --key paper/validation_30/review_key.csv \
        --picks paper/validation_30/round30_picks.csv \
        --adjudication paper/validation_30/adjudication_blind.csv
"""
import argparse
import csv
from pathlib import Path


def norm(v: str) -> str:
    v = (str(v) if v is not None else "").strip().lower()
    if not v:
        return ""
    if v[0] in "abcdef":
        return v[0]
    if "none" in v:
        return "none"
    if "ambig" in v or "unsure" in v:
        return "ambiguous"
    return v[:6]


def read_csv(path: Path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind-adjudicated floor for the N=30 round.")
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path, required=True)
    args = ap.parse_args()

    oracle = {r["row_id"]: norm(r["answer"]) for r in read_csv(args.key)}
    picks = {r["row_id"]: (norm(r["reviewer1"]), norm(r["reviewer2"])) for r in read_csv(args.picks)}
    adj_col = None
    adj_rows = read_csv(args.adjudication)
    if adj_rows:
        adj_col = next((c for c in adj_rows[0] if c.startswith("correct_answer")), None)
    adj = {r["row_id"]: norm(r.get(adj_col, "")) for r in adj_rows} if adj_col else {}

    rids = [r for r in oracle if r in picks]
    right = wrong = ambiguous = 0
    oracle_errors, unfilled = [], []
    for r in rids:
        r1, r2 = picks[r]
        if r1 == r2:                                   # reviewers agree -> human truth, no adjudication
            truth = r1
        else:                                          # clash -> blind-adjudicated truth
            truth = adj.get(r, "")
            if not truth:
                unfilled.append(r)
                continue
        if truth == "ambiguous":
            ambiguous += 1
        elif truth == oracle[r]:
            right += 1
        else:
            wrong += 1
            oracle_errors.append(f"{r} (human={truth or '?'} / oracle={oracle[r]})")

    total = len(rids)
    resolvable = right + wrong
    print(f"cases: {total}   reviewers-agree: {sum(picks[r][0]==picks[r][1] for r in rids)}   clashes: {sum(picks[r][0]!=picks[r][1] for r in rids)}")
    if unfilled:
        print(f"WARNING: {len(unfilled)} clashes not yet adjudicated (fill correct_answer): {unfilled}\n")
    print("=== NUMBERS FOR THE PAPER (confirmatory round) ===")
    if resolvable:
        print(f"  accuracy on resolvable cases : {right}/{resolvable} = {100*right/resolvable:.0f}%  ({ambiguous} ambiguous set aside)")
    print(f"  HONEST FLOOR (ambiguous + unfilled against oracle): {right}/{total} = {100*right/total:.0f}%")
    print(f"  oracle errors: {len(oracle_errors)}" + (("  -> " + "; ".join(oracle_errors)) if oracle_errors else ""))


if __name__ == "__main__":
    main()
