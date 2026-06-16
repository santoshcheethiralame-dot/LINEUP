import argparse

from lineup.backends import Message, TransformersModel
from lineup.config import DEFAULT_MODEL, set_seed

CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
    "between 1887 and 1889."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    set_seed()
    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens)

    messages = [
        Message("system", "Answer the question using only the provided context. Be concise."),
        Message("user", f"Context:\n{CONTEXT}\n\nQuestion: Who is the Eiffel Tower named after?"),
    ]

    generation = model.generate(messages)
    print("answer:", generation.text.strip())
    print(f"generated {len(generation.token_ids)} tokens, total logprob {generation.total_logprob:.3f}")

    scoring = model.score(messages, generation.text)
    print("\nteacher-forced logprobs of the answer:")
    for token, logprob in zip(scoring.tokens, scoring.logprobs):
        print(f"  {token!r:>16}  {logprob:8.3f}")
    print(f"total {scoring.total_logprob:.3f}  mean {scoring.mean_logprob:.3f}")


if __name__ == "__main__":
    main()
