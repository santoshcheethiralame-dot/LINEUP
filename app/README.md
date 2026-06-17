# lineup case explorer

A small Streamlit app that reads the benchmark's saved JSONL — `scenarios`, `generations`, `roles`, `predictions` — and makes the result legible. It does **not** load the model or need a GPU; it is pure data over the files the pipeline emits.

Two views:

- **Overview** — the per-method table (top-1 culprit accuracy, misleading-as-culprit rate, the win-rates), the role-distribution bars, and the selective-answering AUROCs.
- **Case explorer** — pick a case, see its chunks coloured by true role and which chunk each method blamed, with the wrong cases flagged when a method took the near-miss.

## Run locally

```
pip install streamlit
streamlit run app/app.py
```

It opens on a bundled, illustrative sample (`app/sample`). Point the sidebar **data directory** at a real run's `outputs/` to explore actual results.

## Deploy

The only dependency is `streamlit`; the app imports the pure-Python `lineup` modules from `src/`.

- **Streamlit Community Cloud** — connect this repository and set the main file to `app/app.py`.
- **Hugging Face Spaces** — create a Streamlit Space from this repository, set `app_file` to `app/app.py`, and set the Space's `requirements.txt` to `streamlit`.

## Regenerate the sample

```
python app/make_sample.py
```
