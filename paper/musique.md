# MuSiQue cross-dataset check (third dataset, baseline)

| model | cases | wrong | error % | no-culprit % | ContextCite top-1 |
|---|---:|---:|---:|---:|---:|
| phi | 384 | 223 | 58 | 35 | 0.89 |
| qwen | 384 | 219 | 57 | 26 | 0.88 |

The no-culprit headline replicates on MuSiQue, a third independent multi-hop dataset, at
**26-35%** -- in the 27-53% band from HotpotQA/2Wiki -- with ContextCite top-1 still ~0.88.
MuSiQue is markedly harder (error rate ~57-58% vs ~25-30%), yet the ill-posed *fraction* sits
in the same band: ill-posedness is a property of multi-hop RAG error, not of one dataset.