"""Final oracle accuracy from the adjudicated sheet (local, no GPU).

Run after the team has filled the verdict column in adjudication_needed.csv. Combines the uncontested
cases (every rater matched the oracle -> oracle correct) with the team's verdicts on the contested
cases, and prints the two numbers for the paper: accuracy on resolvable cases, and the honest floor
(every ambiguous case counted against the oracle).

    python scripts/score_adjudication.py --key review_key.csv \
        --sheets review_annotator1.csv review_annotator2.csv \
        --adjudication adjudication_needed.csv
"""
import argparse
import csv
from pathlib import Path


def norm(v):
    v = (v or "").strip().lower()
    if v and v[0] in "abcdef":
        return v[0]
    if "none" in v:
        return "none"
    if "unsure" in v or "idk" in v:
        return "unsure"
    return v[:6]


def read_key(path):
    return {(r.get("row_id") or "").strip(): norm(r.get("answer")) for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def read_picks(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rid = (r.get("row_id") or "").strip()
        if rid:
            out[rid] = norm(r.get("your_pick"))
    return out


def read_verdicts(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rid = (r.get("row_id") or "").strip()
        verdict_col = next((c for c in r if c.startswith("verdict")), None)
        v = (r.get(verdict_col) or "").strip().lower() if verdict_col else ""
        if rid:
            if "right" in v:
                out[rid] = "oracle_right"
            elif "wrong" in v:
                out[rid] = "oracle_wrong"
            elif "ambig" in v or "unsure" in v:
                out[rid] = "ambiguous"
            else:
                out[rid] = ""    # not yet filled
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Final adjudicated accuracy + floor.")
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, nargs="+", required=True)
    ap.add_argument("--adjudication", type=Path, required=True)
    args = ap.parse_args()

    key = read_key(args.key)
    sheets = [read_picks(s) for s in args.sheets]
    verdicts = read_verdicts(args.adjudication)
    rids = [r for r in key if key[r] and all(r in s and s[r] for s in sheets)]

    uncontested_right = 0
    contested = []
    for r in rids:
        if all(s[r] == key[r] for s in sheets):
            uncontested_right += 1
        else:
            contested.append(r)

    missing = [r for r in contested if not verdicts.get(r)]
    if missing:
        print(f"WARNING: {len(missing)} contested cases have no verdict yet (fill the verdict column). e.g. {missing[:5]}\n")

    c_right = sum(verdicts.get(r) == "oracle_right" for r in contested)
    c_wrong = sum(verdicts.get(r) == "oracle_wrong" for r in contested)
    c_ambig = sum(verdicts.get(r) == "ambiguous" for r in contested)

    total = len(rids)
    right = uncontested_right + c_right
    wrong = c_wrong
    ambiguous = c_ambig + len(missing)
    resolvable = right + wrong

    print(f"total cases            : {total}")
    print(f"uncontested (all agree): {uncontested_right}  (auto oracle-correct)")
    print(f"contested              : {len(contested)}  ->  oracle_right {c_right}, oracle_wrong {c_wrong}, ambiguous {c_ambig}")
    print()
    print("=== NUMBERS FOR THE PAPER ===")
    if resolvable:
        print(f"  accuracy on resolvable cases : {right}/{resolvable} = {100*right/resolvable:.0f}%   (the {ambiguous} ambiguous set aside)")
    print(f"  HONEST FLOOR (ambiguous count against): {right}/{total} = {100*right/total:.0f}%")
    print(f"  oracle errors: {wrong}/{total} = {100*wrong/total:.0f}%")


if __name__ == "__main__":
    main()
