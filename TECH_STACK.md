# Tech Stack Summary (In Build Order)
### Hallucination Reduction using RAG/RAD for LLM-based QA

Use this as a reference when explaining the project — it walks through every tool/library used, in the order it was introduced, with a one-line reason for each.

---

## 1. Environment
- **Python 3.11.9** — chosen for compatibility, since Python 3.14 (system default) lacked pre-built wheels for the ML libraries needed.
- **venv (virtual environment)** — isolates this project's dependencies from the rest of the system.

## 2. Core Data/ML Libraries
- **numpy (1.26.4, pinned)** — numerical operations underlying nearly every other library.
- **pandas** — used for structuring tabular data (e.g., evaluation results, chart data).
- **tqdm** — progress bars during long-running steps (embedding, dataset loading).
- **torch (CPU build)** — the deep learning framework underlying the LLM and NLI models; CPU-only build specifically chosen since no NVIDIA GPU was available.

## 3. Dataset
- **HuggingFace `datasets` library** — used to download and load the dataset.
- **`rajpurkar/squad` (SQuAD dataset)** — chosen for its clean, separate `question` / `context` / `answers` fields, ideal for building a retrieval-based QA system.

## 4. Retrieval (Phase 2)
- **sentence-transformers** — library providing easy access to embedding models.
- **`all-MiniLM-L6-v2`** — the specific embedding model used to convert text chunks into 384-dimensional vectors representing meaning.
- **faiss-cpu** — Facebook AI's vector similarity search library, used to store chunk embeddings and perform fast nearest-neighbor search for retrieval.

## 5. Generation — Three Pipelines (Phase 3)
- **transformers (HuggingFace)** — library used to load and run the language model.
- **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** — the single LLM used across all three pipelines (chosen for being small enough to run on CPU, while still coherent enough for meaningful comparison, and because it allows direct token-level access needed for RAD).
- **accelerate** — helper library for efficient model loading.

### Pipeline-specific techniques:
- **LLM Baseline** — plain `model.generate()`, no retrieval, no evidence (worst-case hallucination reference point).
- **RAG (Retrieval-Augmented Generation)** — retrieved evidence chunks injected directly into the prompt before generation; still uses `model.generate()`.
- **RAD (Retrieval-Augmented Decoding)** — custom-built manual generation loop (since `model.generate()` can't support this), comparing "with-evidence" vs "without-evidence" predictions at every generated word, boosting evidence-driven word choices. Optimized with:
  - **KV caching** (`past_key_values`) — avoids reprocessing the full sequence at every step.
  - **Repetition penalty** — discourages repeating already-used words, fixing a degeneration issue found during testing.

## 6. Hallucination Detection / Scoring (Phase 4)
- **sentence-transformers `CrossEncoder`** — different mode of the same library, used for pairwise sentence comparison (rather than standalone embeddings).
- **`cross-encoder/nli-deberta-v3-small`** — the Natural Language Inference (NLI) model used to check whether each generated sentence is *supported*, *contradicted*, or *unclear* relative to the retrieved evidence. Chosen over the `base` variant since testing showed both performed identically on hard cases, making `small`'s speed advantage the deciding factor for a CPU-only setup.
- **Custom sentence splitter** (`re`, regex-based) — breaks a full answer into individual sentences before scoring each one separately.
- **Groundedness scoring logic** — custom logic converting NLI labels into a percentage score (supported = full credit, unclear = half credit, contradicted = no credit).

## 7. Evaluation
- **Custom `evaluate.py` script** — runs all three pipelines across a diverse set of test questions, scores each with the NLI scorer, and outputs a summary comparison + saves results to JSON.
- **Hand-labeled validation set** (`nli_evaluation.py`) — a small set of manually-labeled (evidence, claim, true-label) pairs used to measure the NLI scorer's own accuracy (achieved 90% agreement with human judgment on this sample).

## 8. User Interface (Phase 5)
- **Streamlit** — Python-based web app framework, chosen so the UI could be built using the same language and functions already written, without learning a separate frontend stack.
- **Custom CSS (glassmorphism theme)** — injected via `st.markdown(..., unsafe_allow_html=True)` for a dark, frosted-glass visual style.
- **Altair** — charting library used for the color-coded bar charts on the Evaluation Dashboard (chosen over Streamlit's built-in `st.bar_chart` since it supports per-bar custom coloring).
- **`st.session_state`** — Streamlit's built-in state management, used to track "live" results across questions asked within a session for the dynamic dashboard section.
- **`st.tabs()`** — used for instant, no-reload navigation between the Home and Evaluation Dashboard sections (chosen over Streamlit's multi-page `pages/` folder structure, which caused slow full-page reloads).

---

## One-Paragraph Summary (for quick explanations)

*"I built a QA system comparing three approaches to reducing LLM hallucination: a plain LLM baseline, RAG (retrieval-augmented generation), and RAD (retrieval-augmented decoding, a custom-built technique that compares evidence-based vs non-evidence-based word predictions at every generated token). Retrieval uses MiniLM embeddings and FAISS; generation uses a small local LLM (TinyLlama) running entirely on CPU with no paid APIs. Each answer is automatically fact-checked against its evidence using a small NLI (Natural Language Inference) model, producing a groundedness score. Everything is tied together in a Streamlit web app with live and benchmark evaluation dashboards."*
