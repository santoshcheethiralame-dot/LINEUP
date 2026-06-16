from .base import Generation, LanguageModel, Message, Scoring

__all__ = ["Generation", "LanguageModel", "Message", "Scoring", "TransformersModel"]


def __getattr__(name):
    # Import the torch-backed model lazily so that `import lineup.backends` (and the
    # prompt/correctness/generation modules that only need Message/LanguageModel) stay
    # free of a torch dependency.
    if name == "TransformersModel":
        from .transformers_backend import TransformersModel

        return TransformersModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
