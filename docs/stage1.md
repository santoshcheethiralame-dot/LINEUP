# Stage 1 — Data Foundation

## Objective

Stage 1 provides the substrate the rest of the pipeline is built on: a loader that yields, for each question, the gold answer, the gold supporting chunks, and a pool of candidate distractors. The defining requirement is that the **causal chunks are known by construction**. The benchmark's credibility rests on ground-truth chunk roles; if the gold supporting evidence were inferred rather than annotated, every downstream role label would inherit that uncertainty.

## Substrate

The primary source is **HotpotQA** (Yang et al., arXiv:1809.09600) in its *distractor* configuration. HotpotQA is multi-hop: each question requires combining facts from two articles, and the dataset annotates the specific supporting sentences. The distractor configuration additionally supplies, for each question, the two gold paragraphs alongside eight distractor paragraphs, which gives both a known causal core and a realistic set of competing passages out of the box. **2WikiMultiHopQA** (Ho et al., 2020) is retained as a planned second source for a cross-dataset check; the loader is structured so that an additional parser can emit the same schema.

## Chunk Granularity

A chunk is a titled paragraph. This is the unit at which HotpotQA supplies context and at which retrieval naturally operates, so it is the unit at which the benchmark reasons about roles. The annotated supporting facts are sentence-level, so each gold chunk additionally records the indices of its supporting sentences. This finer information is preserved for later stages — in particular the construction of the near-miss chunk in Stage 2, which edits a believable but wrong value inside an otherwise-relevant sentence.

## Gold by Construction

For each question the loader reads the provided paragraphs and the `supporting_facts` annotation. A paragraph that contains at least one supporting sentence is labelled gold; all others are labelled distractors. Each chunk records its provenance — gold or distractor — and, for gold chunks, the supporting sentence indices. Provenance here denotes *where the chunk came from*, and is kept deliberately distinct from the causal *role* assigned later by the oracle: a gold paragraph is part of the intended evidence, which is not the same claim as its having caused any particular answer.

## Schema

Two dataclasses define the interface between this stage and the next. `Chunk` carries an identifier, title, text, the sentence list, provenance, supporting-sentence indices, and an optional retrieval score. `QAExample` carries the question, the gold answer, the gold chunks, the distractor pool, and metadata (question type, difficulty level, source dataset). The parsing of a raw record into a `QAExample` is a pure function, `parse_example`, and is unit-tested against a fixture independently of any dataset download; the loader that reads the dataset is a thin wrapper around it.

## Distractor Retriever

Beyond the distractors HotpotQA already provides, the stage stands up a retriever so that additional, realistic distractors can be drawn from the corpus. The reference implementation is **BM25** (Okapi), chosen for being strong, transparent, and free of heavyweight or platform-specific dependencies. The corpus is the set of paragraphs across the loaded split, de-duplicated by article title so that a single article is not indexed many times. Retrieval tokenizes the question, ranks the corpus by BM25 score, and returns the top *k* paragraphs after excluding any whose title belongs to the question's gold set, so that retrieved distractors are by definition non-gold. Each returned chunk carries its retrieval score for later inspection. A dense retriever (for example, Contriever) implements the same interface and is the planned alternative for a sparse-versus-dense comparison; sparse retrieval is the default here because it is reproducible and dependency-light.

## Output

The stage delivers `load_examples → (question, gold_answer, gold_chunks, candidate_distractors)`, together with a retriever that enriches the distractor pool from the corpus. `scripts/inspect_example.py` demonstrates the output on a single question, printing the gold answer, the gold chunks with their supporting sentences, and the top retrieved distractor candidates.

## Limitations and Planned Extensions

The loader currently targets HotpotQA's distractor configuration; the 2Wiki parser and the dense retriever are deferred but accommodated by the schema and the retriever interface. The corpus for retrieval is scoped to the loaded split for tractability, and the number of examples used to build the index is a parameter, so the trade-off between corpus realism and indexing cost is explicit rather than hidden.

## Relation to the cited work

- **HotpotQA** (Yang et al., arXiv:1809.09600) is the substrate and the source of gold supporting sentences.
- **2WikiMultiHopQA** (Ho et al., 2020) is the planned cross-dataset check.
- **Lewis et al., 2020** (arXiv:2005.11401) and the retrieval survey of **Gao et al.** (arXiv:2312.10997) frame the dense/sparse/hybrid retrieval mechanics behind the distractor retriever.
