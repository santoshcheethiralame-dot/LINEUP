# Running the pipeline (and where a GPU is needed)

The pipeline splits cleanly along compute lines. Only the stages that run the language model need a GPU; everything else is plain CPU work.

| Stage | GPU? | Notes |
| --- | --- | --- |
| 0 infrastructure | only to run the real model | the backend itself runs on CPU with a small model |
| 1 data foundation | no | loaders and the BM25 retriever are CPU-only |
| 2 scenario construction | no | model-free and deterministic |
| 3 generation, 4 oracle, 5 method runners | **yes** | each runs the 7B model many times |
| 6 scorer, write-ups | no | operate on saved outputs |

## Developing without a GPU

Two properties keep development unblocked on a CPU-only machine:

- **Scenario construction is deterministic and model-free.** The machine with the GPU does not need your scenario files; it regenerates the identical cases from the same seed. Code and seed travel through git, and nothing large has to be copied.
- **The backend runs on CPU with a tiny model.** Stage 3–5 *code* can be written and smoke-tested locally against a small model (for example `sshleifer/tiny-gpt2`, which is what the `slow` tests use), then run unchanged on the real model elsewhere. The GPU is only needed for the final run, not for development.

## Running the model stages on a free cloud GPU

A free Colab or Kaggle T4 (16 GB) is enough to run Qwen2.5-7B-Instruct in 4-bit. The notebook [`notebooks/lineup_colab.ipynb`](../notebooks/lineup_colab.ipynb) clones the repository, builds scenarios, and loads the model in 4-bit; open it in Colab, set the runtime to a T4 GPU, and run the cells.

To load the model in 4-bit from code:

```python
from lineup.backends import TransformersModel
from lineup.config import DEFAULT_MODEL

model = TransformersModel(DEFAULT_MODEL, load_in_4bit=True)
```

4-bit (nf4) needs `bitsandbytes` and a CUDA device; it is an optional dependency (`pip install bitsandbytes`, or `pip install -e ".[gpu]"`) and is not part of the CPU install. The compute dtype is fp16 because the T4 has no native bfloat16; the unquantized path makes the same substitution automatically.

## One consistency rule

Run **all** model stages (3–5) on a single machine, against one pinned model revision. Greedy decoding is deterministic for a fixed machine and precision, but log-probabilities drift across different GPUs and quantization settings, so the labels must be produced on one box to be internally comparable. The scenarios are machine-independent, so they may be built anywhere.

## Moving results between machines

Until the scorer (Stage 6), no artifact transfer is required: the GPU machine regenerates the scenarios and runs the model. Once labelled outputs need to be shared back for scoring, the options are a tracked results directory, a Hugging Face dataset, or a shared drive; this is decided when Stage 6 lands.
