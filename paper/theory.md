# Theory scaffold — why single-chunk attribution is ill-posed under redundancy

Formal skeleton for the theory paragraph(s). Definitions + one proposition + the Shapley remark.
Write the prose around this; the claims are stated so they can be checked against the data
(dose-response Fig 7 and recovery Fig 2 are the empirical mirror of the bound below).

## Setup

- A case is a question `q` with retrieved context `C = {c_1, ..., c_k}`. A model `M` returns an
  answer `M(q, S)` for any subset `S ⊆ C`. The observed answer is `a = M(q, C)`; the error is `a ≠ a*`.
- Write `val(x)` for the canonical value an answer string carries (the oracle's `answer_key`), so
  "the designer was X" and "X" share a value. Two answers are equal iff their values are equal.

## Definitions

- **Necessity (the leave-one-out oracle).** Chunk `c_i` is *necessary* for `a` if removing it
  changes the value: `val(M(q, C \ {c_i})) ≠ val(a)`. This is exactly the oracle's `causal` axis.
- **Salience.** `c_i` is *salient* if it asserts `val(a)` (the oracle's `salient` axis).
- **Culprit** = necessary ∧ salient. **Misleading** = salient ∧ ¬necessary.
- **Sufficient cause.** A set `S ⊆ C` is *sufficient* for `a` if `val(M(q, S)) = val(a)`.
  `S` is a *minimal* sufficient cause if no proper subset is sufficient.
- **m-fold redundancy.** There exist `m ≥ 2` chunks each of which is *alone sufficient* for the
  wrong value `w`: `val(M(q, {c_i})) = w` for each of the `m`. (Our hard-traps construction plants
  exactly this: the near-miss plus `m-1` decoys, all asserting `w`.) Call this set the **responsible
  set** `R`, `|R| = m`.

## Proposition (single-chunk attribution is ill-posed under redundancy)

Assume `m`-fold redundancy with `m ≥ 2` and that the model adopts the redundant value
(`val(a) = w`). Then:

1. **No chunk is necessary.** For any `c_i ∈ R`, the other `m-1` copies still force `w`, so
   `val(M(q, C \ {c_i})) = w = val(a)`; hence `c_i` is not necessary. The leave-one-out oracle
   therefore assigns **no culprit** — every member of `R` is labelled *misleading*. *(This is the
   "no single culprit" regime, measured at 27–53% of organic errors.)*

2. **The minimal sufficient causes are the `m` singletons** `{c_i}`, `c_i ∈ R` — none necessary,
   none distinguished from the others. Which one to "blame" is underdetermined by the data.

3. **Top-1 recall is bounded.** Any predictor that returns a single chunk recovers at most a
   `1/m` fraction of `R`:  `recall@1 = |{top-1} ∩ R| / |R| ≤ 1/m`.  As redundancy grows, top-1
   recall → 0, while a size-`m` set can recover all of `R` (`recall@m = 1`). *(Empirical mirror:
   recall@1 ≈ 0.3 at `m ≈ 2`; the dose-response sweep traces the `1/m` decay.)*

**Proof sketch.** (1) and (2) are immediate from the definitions of necessity and sufficiency under
`m`-fold redundancy. (3): `{top-1}` has one element, so `|{top-1} ∩ R| ≤ 1`, giving
`recall@1 ≤ 1/|R| = 1/m`; the effect-set of the `m` highest-effect chunks attains `recall = 1` when
the method ranks `R` above the rest. ∎

## Proposition 2 (Shapley attribution dilutes redundant credit to 1/m)

Model the wrong value as a monotone-OR game: `v(S) = 1` if `val(M(q, S)) = w` else `0`, and assume
the model produces `w` iff `S` contains at least one member of `R` (any single responsible chunk
suffices — exactly our redundancy construction). Then the Shapley value `φ_i` satisfies

- `φ_i = 1/m` for every `c_i ∈ R`, and `φ_j = 0` for every `c_j ∉ R`.

**Proof.** Non-members are *null players*: for any `S`, `v(S ∪ {c_j}) = v(S)` when `c_j ∉ R` (adding a
non-responsible chunk never changes whether some responsible chunk is present), so `φ_j = 0`. The `m`
members of `R` are mutually *symmetric* in `v` (interchangeable), so Shapley symmetry gives them equal
value. Efficiency gives `Σ_i φ_i = v(C) − v(∅) = 1 − 0 = 1`. Equal shares summing to 1 over `m`
players give `φ_i = 1/m`. ∎

So Shapley reports each responsible chunk as only `1/m` responsible, vanishing as redundancy grows, and
never expresses that each is *alone sufficient* or that *no single chunk is necessary*. The
no-single-culprit structure is invisible to any per-chunk scalar score — single-chunk pick or Shapley
alike — because it is a property of the *set* of sufficient causes, not of any one chunk. This is the
formal case for set-valued attribution.

## Proposition 3 (a conformal set restores a coverage guarantee)

The remedy lives at the set level. Let `s(c)` be any per-chunk score and, for a labelled case, let the
nonconformity score be the rank of the true culprit under `s` (1 = top). Given an exchangeable
calibration set of size `n` and miscoverage `α`, let `τ_α` be the `⌈(1−α)(n+1)⌉`-th smallest
calibration rank, and output the top-`τ_α` chunks. By the standard split-conformal guarantee
[Vovk et al.; Angelopoulos & Bates], for an exchangeable test case

- `P(culprit ∈ top-τ_α set) ≥ 1 − α`,

distribution-free, assuming only exchangeability — no model of `s` or of the data. *(Empirical mirror:
α=0.1 gives `τ=2` and test coverage 0.94; the calibrated `τ` rises 2→3 from baseline to hard-traps, so
the set size itself reports the redundancy of Proposition 1.)* We invoke the standard result rather than
reprove it; the contribution is applying it to attribution and observing the size–redundancy link.

## Takeaways to state in the paper
- Necessity-based attribution (leave-one-out, ContextCite's ablation target) is *undefined* as a
  single-chunk answer whenever the cause is redundant — not merely inaccurate.
- The right object is the (minimal) sufficient-cause **set**; the right action when it is not a
  singleton is to **abstain** from a single answer. Both are what our remedy delivers.
