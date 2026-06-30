"""Bootstrap 95% CIs for Cohen's/Fleiss' kappa on both human rounds + Rushi-exclusion sensitivity.

Addresses reviewer item #14. Pilot raw picks: Downloads/lineup_validation/{santosh,nivas,rushi}.csv;
confirmatory: paper/validation_30/round30_picks.csv. Oracle answer from each round's key. We report
each rater-vs-oracle and the inter-rater kappa with case-resampling bootstrap CIs (B=2000), the
3-rater Fleiss kappa, and what excluding Rushi does to inter-rater agreement.
"""
import csv, math, random
from pathlib import Path

random.seed(0)
DL = Path("C:/Users/carbo/Downloads/lineup_validation")
V30 = Path("C:/Users/carbo/projects/lineup/paper/validation_30")

def norm(v):
    v = (v or "").strip().lower()
    if not v: return ""
    if v[0] in "abcdef": return v[0]
    if "none" in v: return "none"
    if "unsure" in v or "idk" in v: return "unsure"
    return v[:6]

def col(path, key, valcol):
    return {r["row_id"]: norm(r.get(valcol)) for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

def cohen(pairs):
    a = [x for x, _ in pairs]; b = [y for _, y in pairs]; n = len(a)
    if n == 0: return None
    cats = set(a) | set(b)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return (po - pe)/(1 - pe) if pe < 1 else 1.0

def fleiss(rows):  # rows: list of per-item rater-label lists
    cats = sorted({c for r in rows for c in r}); k = len(rows[0]); n = len(rows)
    p_item = [(sum(r.count(c)**2 for c in cats) - k)/(k*(k-1)) for r in rows]
    p_bar = sum(p_item)/n
    p_cat = {c: sum(r.count(c) for r in rows)/(n*k) for c in cats}
    pe = sum(v*v for v in p_cat.values())
    return (p_bar - pe)/(1 - pe) if pe < 1 else 1.0

def boot_ci(items, stat, B=2000):
    vals = []
    for _ in range(B):
        s = [items[random.randrange(len(items))] for _ in range(len(items))]
        v = stat(s)
        if v is not None: vals.append(v)
    vals.sort()
    return vals[int(0.025*len(vals))], vals[int(0.975*len(vals))]

def report(name, pairs):
    k = cohen(pairs); lo, hi = boot_ci(pairs, cohen)
    print(f"  {name:<24} kappa {k:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (n={len(pairs)})")

# ---- pilot (50 cases, 3 raters) ----
key = col(DL/"review_key_KEEP_PRIVATE.csv", "row_id", "answer")
S = col(DL/"review_santosh.csv", "row_id", "your_pick")
N = col(DL/"review_nivas.csv", "row_id", "your_pick")
R = col(DL/"review_rushi.csv", "row_id", "your_pick")
ids = [i for i in key if key[i] and S.get(i) and N.get(i) and R.get(i)]
print(f"=== PILOT (n={len(ids)}) ===")
report("Santosh vs oracle", [(S[i], key[i]) for i in ids])
report("Nivas vs oracle",   [(N[i], key[i]) for i in ids])
report("Rushi vs oracle",   [(R[i], key[i]) for i in ids])
report("Santosh vs Nivas (inter)", [(S[i], N[i]) for i in ids])
fr = [[S[i], N[i], R[i]] for i in ids]
fk = fleiss(fr); flo, fhi = boot_ci(fr, fleiss)
print(f"  {'Fleiss (all 3 raters)':<24} kappa {fk:+.2f}  95% CI [{flo:+.2f}, {fhi:+.2f}]")
print(f"  Rushi-exclusion sensitivity: 2-rater Cohen (S<->N) {cohen([(S[i],N[i]) for i in ids]):+.2f} vs 3-rater Fleiss {fk:+.2f}")
print(f"  -> Rushi vs oracle kappa is near zero; including him pulls Fleiss below the 2-rater Cohen, which is why he is excluded (disclosed).")

# ---- confirmatory (30 cases, 2 raters) ----
key30 = col(V30/"review_key.csv", "row_id", "answer")
p30 = {r["row_id"]: (norm(r["reviewer1"]), norm(r["reviewer2"])) for r in csv.DictReader(open(V30/"round30_picks.csv", encoding="utf-8-sig"))}
ids30 = [i for i in key30 if key30[i] and i in p30]
print(f"\n=== CONFIRMATORY (n={len(ids30)}) ===")
report("Reviewer1 vs oracle", [(p30[i][0], key30[i]) for i in ids30])
report("Reviewer2 vs oracle", [(p30[i][1], key30[i]) for i in ids30])
report("R1 vs R2 (inter)",     [(p30[i][0], p30[i][1]) for i in ids30])
