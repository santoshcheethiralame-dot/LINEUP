# Are wrong attributions salience-shaped?

Well-posed cases only (a single ground-truth culprit exists). For each method we take the
cases where its top pick is not the culprit, and report the role of the chunk it blamed.
The *misleading* chunk is the pure salience trap. 'Chance' is the rate a uniform wrong pick
would hit a misleading chunk (≈ #misleading / #non-culprit per case).

| method | wrong picks | → misleading (salient) | chance | lift | → inert | → silent |
|---|---:|---:|---:|---:|---:|---:|
| contextcite | 195 | 50.8% | 20.3% | 2.50x | 17.9% | 31.3% |
| single_chunk | 238 | 59.7% | 21.2% | 2.82x | 26.1% | 14.3% |
| lexical_similarity | 707 | 13.9% | 12.6% | 1.10x | 58.4% | 27.7% |
| llm_judge | 336 | 37.5% | 15.5% | 2.42x | 43.8% | 18.8% |

Reading: a lift >1 on 'misleading' means the method's *errors* are biased toward the salient
red-herring beyond chance -- 'not fooled overall, but errors are salience-shaped'. The effect-
based methods (contextcite, single_chunk) and llm_judge are strongly trap-biased (2.4-2.8x):
they confuse the salient near-twin of the culprit for the culprit. Lexical is the surprise --
its errors are NOT trap-biased (1.1x); they scatter toward inert chunks. So salience-shaped
failure is a property of the causal methods, not of surface-overlap scoring.