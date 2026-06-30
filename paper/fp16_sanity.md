# 4-bit vs fp16 quantization sanity (reviewer item #16)

Re-ran the full generate -> leave-one-out pipeline in **fp16** (float16, the 4-bit compute dtype) on a
matched **Qwen-2.5-7B** subsample: 60 wrong@4bit cases per cell x 4 cells (both datasets, both
regimes), comparing against the shipped 4-bit labels. Compared on the cases wrong under *both*
precisions. Run on Kaggle 2xT4 (fp16 7B sharded across the pair). Script:
`scripts/run_fp16_sanity.py` (salient-only ablation -- identical has-culprit label as the full oracle,
verified 0/1766 mismatches offline).

| cell | n_both | no-culprit 4-bit | no-culprit fp16 | per-case agree |
|---|--:|--:|--:|--:|
| hotpotqa/baseline/qwen  | 46 | 28% | 20% | 87% |
| hotpotqa/hardtraps/qwen | 50 | 46% | 46% | 84% |
| 2wiki/baseline/qwen     | 51 | 27% | 35% | 88% |
| 2wiki/hardtraps/qwen    | 53 | 28% | 23% | 91% |
| **pooled**              | **200** | **32%** | **31%** | **88%** |

Of 240 wrong@4bit cases, 200 (83%) remain wrong in fp16 (precision shifts a few answers, expected);
the comparison is on those 200. Pooled no-single-culprit rate is **31% (fp16) vs 32% (4-bit)** -- a
1-point difference -- with **88%** per-case culprit/no-culprit agreement. The hardtraps cells (the
redundant-coalition regime) agree most tightly (46/46, 28/23). **The no-single-culprit finding is not
a quantization artifact.**

Note: these are Qwen-only, matched-subsample rates (n_both=200), distinct from the all-cell headline
no-culprit band (strict 43% / shipped 37%, n=1766). The claim here is precision *agreement* on matched
cases, not reproduction of the headline. Phi-3.5 / Mistral-7B reproduce by swapping `--model` + paths.
