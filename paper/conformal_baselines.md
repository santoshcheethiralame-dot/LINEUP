# Conformal vs naive set baselines (honest comparison)

Well-posed test cases: **390**; target coverage 1-alpha = 90%.

| set rule | coverage | avg size | guarantee |
|---|---:|---:|---|
| single pick (top-1) | 0.82 | 1.00 | uncalibrated |
| fixed top-2 | 0.94 | 2.00 | uncalibrated |
| score-threshold (>=0.5*top) | 0.94 | 1.51 | uncalibrated (tuned frac) |
| split-conformal (tau=2) | 0.94 | 2.00 | GUARANTEED >= 1-alpha |

Naive rules reach comparable coverage on this data, so conformal is **not more efficient**.
Its contribution is the **distribution-free guarantee**: only conformal hits a *chosen* level
by construction. The fixed-k and threshold rules give whatever coverage they give — their
number depends on a hand-tuned k/threshold, with no promise it holds at another level or off
this distribution. This is the right way to motivate conformal (calibration, not performance),
and it pre-empts the obvious 'why not just top-k?' question.

**Per condition (does it adapt?):** fixed top-2 covers 0.94 (baseline) and 0.93 (hard-traps) — it
does *not* collapse under redundancy; conformal simply keeps a safer margin by growing tau 2->3 (coverage 0.94->0.96).

**Negative result:** an adaptive per-case APS set bloats (coverage 0.91, avg size 3.63) because ContextCite's effect scores are diffuse, so a fixed-tau conformal set
is preferable. (Adaptive sizing is future work, perhaps with a less diffuse base scorer.)