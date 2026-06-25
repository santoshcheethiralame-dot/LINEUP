# ContextCite ablation-count sensitivity (HotpotQA/qwen baseline)

| N ablations | top-1 culprit acc | recall@1 | recall@k | n |
|---:|---:|---:|---:|---:|
| 8 | 0.84 | 0.33 | 0.33 | 85 |
| 16 | 0.93 | 0.37 | 0.37 | 85 |
| 32 (reference) | 0.92 | 0.41 | 0.41 | 64 |
| 64 | 0.94 | 0.38 | 0.38 | 85 |

Top-1 culprit accuracy is **stable for N>=16** (0.92-0.94, spread 0.02); it dips at
N=8 (0.84) where there are too few ablation samples for the Lasso. The default **N=32 sits in the
stable regime**, so the headline results do not hinge on the ablation count provided it is not set
pathologically low — closing the stated limitation. (The N=32 row is the main-matrix baseline, a
separate smaller sample; the N=8/16/64 sweep shares one 85-case set.) (Fig 15, appendix.)