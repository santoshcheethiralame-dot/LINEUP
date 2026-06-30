"""Score the filled blind-validation sheets and emit the adjudication sheet (local, no GPU).

Run this once both annotators have filled their copies (the your_pick column = a letter / NONE /
UNSURE). It prints the numbers that go straight into the paper -- each annotator vs the oracle, the
inter-rater Cohen's kappa (Fleiss if 3+ raters), and the culprit-vs-no-culprit agreement split -- and
writes adjudication_needed.csv listing only the contested cases (some annotator disagrees with the
oracle). Fill the verdict column in that file together, then run score_adjudication.py for the final
accuracy and the honest floor.

    python scripts/run_validation_kappa.py --key review_key.csv \
        --sheets review_annotator1.csv review_annotator2.csv
"""
import argparse
import csv
from itertools import combinations
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
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[(r.get("row_id") or "").strip()] = norm(r.get("answer"))
    return out


def read_sheet(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        rid = (r.get("row_id") or "").strip()
        if rid:
            out[rid] = {
                "pick": norm(r.get("your_pick")),
                "reason": (r.get("reason") or "").strip(),
                "question": (r.get("question") or "").strip(),
                "model_answer": (r.get("model_answer") or "").strip(),
            }
    return out


def cohen(a, b):
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return po, ((po - pe) / (1 - pe) if pe < 1 else 1.0)


def fleiss(rows):
    cats = sorted({c for row in rows for c in row})
    k, n = len(rows[0]), len(rows)
    p_item = [(sum(row.count(c) ** 2 for c in cats) - k) / (k * (k - 1)) for row in rows]
    p_bar = sum(p_item) / n
    p_cat = {c: sum(row.count(c) for row in rows) / (n * k) for c in cats}
    p_e = sum(v * v for v in p_cat.values())
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation agreement + adjudication sheet.")
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None, help="adjudication sheet path")
    args = ap.parse_args()

    key = read_key(args.key)
    sheets = {s.stem: read_sheet(s) for s in args.sheets}
    names = list(sheets)
    rids = [r for r in key if key[r] and all(r in sheets[n] and sheets[n][r]["pick"] for n in names)]
    if not rids:
        print("No rows where every rater answered. Check the your_pick column is filled and row_ids match.")
        return
    print(f"cases scored (every rater answered): {len(rids)}\n")

    print("=== AGREEMENT WITH THE ORACLE  (report these) ===")
    for n in names:
        po, k = cohen([sheets[n][r]["pick"] for r in rids], [key[r] for r in rids])
        print(f"  {n:16s} vs oracle : raw {po:.2f}   Cohen kappa {k:.2f}")
    print("\n=== INTER-RATER  (the human ceiling) ===")
    for x, y in combinations(names, 2):
        po, k = cohen([sheets[x][r]["pick"] for r in rids], [sheets[y][r]["pick"] for r in rids])
        print(f"  {x} <-> {y} : raw {po:.2f}   Cohen kappa {k:.2f}")
    if len(names) >= 3:
        print(f"  Fleiss kappa (all {len(names)} raters): {fleiss([[sheets[n][r]['pick'] for n in names] for r in rids]):.2f}")

    print("\n=== WHERE THE DISAGREEMENT LIVES  (the thesis: humans force a pick when there is none) ===")
    cul = [r for r in rids if key[r] != "none"]
    non = [r for r in rids if key[r] == "none"]
    for n in names:
        ac = sum(sheets[n][r]["pick"] == key[r] for r in cul) / len(cul) if cul else 0.0
        an = sum(sheets[n][r]["pick"] == key[r] for r in non) / len(non) if non else 0.0
        print(f"  {n:16s} agrees with oracle: culprit cases {ac:.2f}  vs  no-culprit cases {an:.2f}")
    print(f"  (n = {len(cul)} culprit, {len(non)} no-culprit)")

    contested = [r for r in rids if any(sheets[n][r]["pick"] != key[r] for n in names)]
    out = args.out or (args.key.parent / "adjudication_needed.csv")
    cols = (["row_id", "question", "model_answer", "oracle_pick"]
            + [f"{n}_pick" for n in names] + [f"{n}_reason" for n in names]
            + ["verdict (oracle_right / oracle_wrong / ambiguous)"])
    ctx = sheets[names[0]]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in contested:
            w.writerow([r, ctx[r]["question"], ctx[r]["model_answer"], key[r]]
                       + [sheets[n][r]["pick"] for n in names]
                       + [sheets[n][r]["reason"] for n in names] + [""])

    print("\n=== NEXT STEP ===")
    print(f"{len(rids) - len(contested)} cases: every rater matched the oracle -> auto oracle-correct.")
    print(f"{len(contested)} contested cases written to:\n  {out}")
    print("Fill the last column (verdict) together, then run score_adjudication.py.")


if __name__ == "__main__":
    main()
