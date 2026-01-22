# Repository Guidelines

## Project Structure & Module Organization
- `app.py` runs the Streamlit demo UI for the RAG system.
- `build_vector_store.py` builds the Chroma vector store (default output `./chroma_rag_db`).
- `eval.py` evaluates baseline vs RAG and prints accuracy/hallucination rates.
- `config.py` holds dataset paths and evaluation settings.
- `dataset/` contains the cMedQA2 data (CSV + candidates lists).
- `learning_content/` contains experiments, demos, and sample Chroma DBs; treat as reference material.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies.
- `python build_vector_store.py` creates the vector store used by retrieval.
- `python eval.py` runs evaluation and prints metrics to stdout.
- `streamlit run app.py` starts the local web demo.

## Coding Style & Naming Conventions
- Python only; use 4-space indentation and PEP 8 style where practical.
- Prefer clear, descriptive names (e.g., `build_vector_store.py`, `topk_scan.py`).
- Keep paths configurable via `config.py` rather than hardcoding in scripts.
- No formatter or linter is configured; keep changes minimal and readable.

## Testing Guidelines
- No automated test suite is present. Validate changes by running:
  - `python eval.py` for metrics sanity checks.
  - `streamlit run app.py` for UI smoke testing.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative subjects like `Add ...` or `Update ...`.
- Keep commits small and focused; include the script or data area you touched.
- PRs should describe dataset changes, model/embedding assumptions, and any new
  generated artifacts (e.g., `chroma_rag_db`).

## Configuration & Data Notes
- Update `config.py` to match your local dataset paths before running.
- The app expects an Ollama server at `http://localhost:11434` and local models
  for chat and embeddings; document any deviations in your PR description.
