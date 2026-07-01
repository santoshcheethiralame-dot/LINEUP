# Position-controlled ablation (reviewer R1) — content, not position

Leave-one-out *deletion* shifts every later passage one slot earlier and shortens the context, so a
position-sensitive model could flip its answer for a positional (not content) reason and inflate the
causal axis. We re-ran the causal test with **length-matched masking**: each ablated passage stays in
its slot `[i]` but its title+body are replaced by content-free filler of equal token length, holding
every other passage's position and the total context length fixed. Matched Qwen-2.5-7B subsample,
4-bit, 60 wrong cases/cell. Script: `scripts/run_position_control.py`.

| cell | n | no-culprit (del) | no-culprit (mask) | agree | CC acc (del) | CC acc (mask) |
|---|--:|--:|--:|--:|--:|--:|
| hotpotqa/baseline/qwen  | 60 | 30% | 28% | 98% | 93% | 93% |
| hotpotqa/hardtraps/qwen | 60 | 42% | 42% | 93% | 89% | **77%** |
| 2wiki/baseline/qwen     | 60 | 25% | 25% | 97% | 87% | 91% |
| 2wiki/hardtraps/qwen    | 60 | 27% | 32% | 95% | 91% | 88% |
| **POOLED**              | **240** | **31%** | **32%** | **96%** | **90%** | **88%** |

**Masking reproduces deletion.** The no-single-culprit rate moves 31%→32% (1 point), the per-case
culprit/no-culprit label agrees in **96%** of cases, and ContextCite culprit-accuracy is 88% vs 90%.
The causal axis is a property of passage **content**, not position — the R1 confound does not drive
the finding, and the no-single-culprit headline survives a position-controlled ablation.

**Honest caveat (one cell):** in HotpotQA/hard-traps, ContextCite culprit-accuracy drops 89%→77%
under masking, while the no-culprit rate is *identical* (42%/42%). Ablation-based *localization*
inherits some of the model's position sensitivity even where the aggregate no-culprit finding does
not — worth a sentence, not a threat to the headline.

Consistency: live-recomputed deletion (31%) matches the shipped 4-bit labels for these Qwen cells
(~29–32%) and the fp16 sanity (4-bit 32% / fp16 31%) — three independent computations agree.
