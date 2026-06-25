# Real-retriever slice (HotpotQA/qwen)

| slice | wrong | no-culprit % | ContextCite top-1 |
|---|---:|---:|---:|
| real-retriever (BM25 distractors) | 104 | 31 | 0.96 |
| natural (dataset distractors) | 99 | 28 | 0.96 |
| constructed (planted near-miss) | 90 | 29 | 0.92 |

With a **real BM25 retriever's** distractors -- the hardest, most lexically-similar passages over
the corpus -- the no-culprit rate matches the dataset-distractor and constructed slices. The
ill-posedness is not an artifact of the dataset's curated distractors; it holds under genuine
retrieval. Closes the 'your context isn't a real retrieval' objection.