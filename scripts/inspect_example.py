import argparse

from lineup.data.hotpotqa import load_examples
from lineup.data.retrieval import BM25DistractorRetriever, build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index-size", type=int, default=500, help="examples used to build the BM25 corpus")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    examples = list(load_examples(args.split, limit=args.index_size))
    retriever = BM25DistractorRetriever(build_corpus(examples))

    example = examples[0]
    print(f"Q: {example.question}")
    print(f"A: {example.answer}\n")

    print("gold chunks:")
    for chunk in example.gold_chunks:
        print(f"  [{chunk.title}] supporting sentences {chunk.supporting_sentence_ids}")
        for sentence_id in chunk.supporting_sentence_ids:
            print(f"      - {chunk.sentences[sentence_id].strip()}")

    print("\nretrieved distractor candidates:")
    gold_titles = [chunk.title for chunk in example.gold_chunks]
    for chunk in retriever.retrieve(example.question, k=args.k, exclude_titles=gold_titles):
        print(f"  [{chunk.title}]  score={chunk.retrieval_score:.2f}")


if __name__ == "__main__":
    main()
