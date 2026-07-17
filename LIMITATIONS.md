# Project Limitations Log
### Hallucination Reduction using RAG/RAD for LLM-based QA

This document tracks every limitation encountered during the project, split into two categories:
1. **Resolved** — issues we hit and fixed, with how
2. **Unresolved / Documented** — real limitations we identified, understood, but did not fix, with the reason why

---

## 1. Resolved Limitations

### 1.1 Dataset field confusion (early project attempt)
**Issue:** Original dataset (`sentence-transformers/natural-questions`) only had `query` and `answer` fields — no separate `context`/`passage` field, causing incorrect chunking logic.
**Fix:** Switched to `rajpurkar/squad`, which has clean, separate `question`, `context`, and `answers` fields.

### 1.2 Python version incompatibility
**Issue:** System had Python 3.14 installed, which lacked pre-built wheels for `torch`, `faiss-cpu`, and other ML libraries.
**Fix:** Installed Python 3.11.9 alongside 3.14, created the project's virtual environment specifically with `py -3.11 -m venv venv`.

### 1.3 Numpy version drift
**Issue:** Installing `accelerate` silently upgraded `numpy` from 1.26.4 to 2.4.6, risking compatibility issues with `faiss-cpu`/`torch`.
**Fix:** Re-pinned numpy to 1.26.4 and verified the full pipeline (retrieval, embeddings) still worked correctly afterward.

### 1.4 Duplicate context chunking
**Issue:** Many SQuAD questions share the same underlying paragraph; naive chunking would have re-chunked identical paragraphs multiple times.
**Fix:** Deduplicated contexts using `set()` before chunking — reduced 2000 examples down to 266 unique paragraphs before creating 759 chunks.

### 1.5 RAD generation speed (initial)
**Issue:** RAD requires two forward passes per generated word (with-evidence and without-evidence), making it far slower than RAG/baseline, especially on CPU with no optimization.
**Fix:** Added KV caching (`past_key_values`, `use_cache=True`) so only the newest token is processed each step instead of reprocessing the entire growing sequence — significant speedup with zero quality loss.

### 1.6 Evaluation topic mismatch
**Issue:** `evaluate.py` initially selected diverse test questions from the *full* 87,599-example SQuAD dataset, while the FAISS retrieval index only contained chunks from the first 2000 examples. This caused irrelevant topics (e.g., iPod, Zelda questions) to retrieve unrelated evidence (e.g., Beyoncé chunks), producing nonsensical answers and misleading scores.
**Fix:** Restricted diverse-question sampling to the same first 2000 examples used to build the index (`dataset["train"].select(range(2000))`), ensuring test questions always have real, relevant evidence available.

### 1.7 RAD repetition / degeneration
**Issue:** RAD's raw greedy decoding loop (built manually, since `model.generate()` couldn't support the dual with/without-evidence comparison) occasionally repeated phrases verbatim within a single answer (e.g., "a career spanning 19 years" appearing twice).
**Fix:** Added a repetition penalty inside the manual generation loop — tokens already used in the current answer have their scores reduced before the next token is chosen. Confirmed fix via before/after test on the same question (score improved from 75% to 100%, repetition eliminated).

### 1.8 RAD answers cut off mid-sentence
**Issue:** RAD was capped at `max_new_tokens=50` while RAG/baseline used 100, causing RAD's answers to sometimes be truncated mid-thought, unfairly penalizing its groundedness score.
**Fix:** Raised RAD's `max_new_tokens` to 100 to match the other two pipelines, ensuring a fair, apples-to-apples comparison.

### 1.9 VS Code / Streamlit environment mismatch
**Issue:** `streamlit run app.py` was launching using a different Python installation than the project's venv, causing `ModuleNotFoundError: No module named 'transformers'` even though transformers was correctly installed in the venv.
**Fix:** Reinstalled Streamlit explicitly inside the venv, and launched using `python -m streamlit run app.py` to guarantee the correct interpreter is used.

### 1.10 Slow page navigation in UI
**Issue:** Using Streamlit's multi-page (`pages/` folder + `st.switch_page()`) structure caused full page reloads on every navigation, which would have required re-loading all models (TinyLlama, embeddings, NLI) on every single tab switch once real logic was wired in.
**Fix:** Restructured the UI into a single-file app using `st.tabs()` instead of separate pages — tab switching is now instant since it doesn't trigger a script/model reload.

---

## 2. Unresolved / Documented Limitations

### 2.1 NLI cannot catch false-equivalence claims
**Issue:** A claim like *"The statue of Mary on the Main Building is the Notre Dame Grotto"* incorrectly equates two separate, individually-true facts. Both our small (`nli-deberta-v3-small`) and base (`nli-deberta-v3-base`) NLI models scored this as "supported," since both halves of the claim are individually present in the evidence.
**Why unresolved:** Tested both model sizes — the error persisted identically in both, ruling out model capacity as the cause. This is a structural blind spot of general-purpose NLI: it checks whether stated content is supported, not whether two supported facts are wrongly claimed to be identical. A rule-based patch (checking word proximity in evidence) was attempted but proved unreliable (it grabbed connector words like "Dame" instead of the true subject "statue," giving false negatives). Properly solving this would require named entity recognition or dependency parsing — beyond this project's scope.

### 2.2 NLI is overly strict about exact phrasing
**Issue:** Near word-for-word claims can still be scored "unclear" or even "contradicted" if they include minor additional detail not verbatim in the evidence sentence (e.g., adding "at Notre Dame" to an otherwise correct claim flipped the model's judgment away from "supported").
**Why unresolved:** Confirmed via isolated testing (identical text scored "supported," but the same text plus 2 extra words did not) that this is genuine model behavior, not a code bug. This is a known characteristic of smaller NLI models trained on relatively short, tightly-matched sentence pairs.

### 2.3 Sentence-level scoring is too coarse for long/run-on sentences
**Issue:** When a model (particularly RAD) generates one long, multi-clause sentence mixing correct and incorrect details, the entire sentence gets one single label — a single wrong or garbled word can drag the whole sentence's score down, even if 90% of it is accurate.
**Why unresolved:** Splitting only on `.`/`!`/`?` is simple and reliable, but going further (e.g., splitting on commas/clauses) risks breaking sentences at points that aren't really separate claims, adding more complexity than value. Documented as a known coarseness tradeoff.

### 2.4 Scorer has a bias toward short, verbatim-style answers
**Issue:** RAG tends to produce short, evidence-mirroring sentences that often score well, while RAD tends to produce longer, more detailed, well-formed paraphrases that frequently score lower — even when RAD's answer is, on manual inspection, more complete or equally/more correct.
**Why unresolved:** This is the compounded effect of limitations 2.2 and 2.3 — the scorer rewards brevity and close wording, not necessarily correctness. This means the numeric groundedness score should be read as a **directional signal**, not a precise correctness measure. Confirmed as a consistent pattern across multiple live test questions, not a one-off.

### 2.5 RAD is not universally more accurate than RAG on every question
**Issue:** While RAD's *average* groundedness score across the evaluation set was higher than RAG's (58.3% vs 50.0%), on individual questions RAD sometimes underperformed RAG — e.g., the antibiotics question, where RAD confidently mixed up "antibiotics" with "anti-inflammatory drugs" (aspirin/ibuprofen), a genuine content error rather than a scoring artifact.
**Why unresolved:** This reflects a real, honest limitation of RAD's evidence-weighting mechanism — it nudges word choice toward evidence-supported vocabulary, but doesn't fully prevent the underlying LLM from drifting into confused/adjacent domain knowledge not explicitly ruled out by the evidence. The core hypothesis (RAD improves grounding *on average*) still held, but it is not a guarantee on every single question — expected given TinyLlama's small size and the small evaluation sample.

### 2.6 NLI-based scoring alternatives were not implemented, due to CPU constraints
**Issue:** A constrained LLM-judge (asking a full LLM a narrow yes/no grounding question, rather than a small NLI classifier) likely would have handled RAD's longer, well-formed, paraphrased answers more fairly than our small NLI model.
**Why unresolved:** This project is CPU-only with no GPU access. RAD's generation is already the slowest part of the pipeline; adding a full LLM-based scorer (run on every sentence, across all three pipelines) would have significantly compounded runtime. This is flagged as a clear, reasonable direction for future work with more compute available, not attempted here.

### 2.7 Evaluation sample size is small
**Issue:** Final evaluation runs used 6–10 questions, not hundreds, due to RAD's slow generation on CPU (roughly several minutes per full 3-pipeline run per question).
**Why unresolved:** A CPU-only, time-constrained environment made larger-scale evaluation (50+ questions) impractical within the project's scope. The observed trend (Baseline < RAG < RAD) is consistent across two separate small-sample runs, but should be read as suggestive, not statistically robust. Noted as a natural next step if more compute/time is available.

### 2.8 UI page-switch cosmetic flash (partially mitigated)
**Issue:** Even after moving to tab-based navigation, and previously with page-based navigation, brief visual flashes/default styling occasionally appeared during Streamlit reruns.
**Why unresolved:** This is a platform-level rendering characteristic of Streamlit (CSS injection happens after each script rerun) rather than something fixable in application code. A loading spinner was added as a partial cosmetic mitigation; the underlying behavior is a known Streamlit limitation.

### 2.9 Persistent CSS styling issue on tab-highlight underline
**Issue:** A red underline artifact (Streamlit's default active-tab indicator) persisted beneath the custom-styled "Home" tab despite multiple targeted CSS override attempts (`data-baseweb`, `data-testid`, `::before`/`::after`, universal child selectors).
**Why unresolved:** Streamlit's internal component library renders this indicator through mechanisms that weren't fully identified even via DOM inspection; further attempts risked diminishing returns relative to the cosmetic impact. Documented as a known, minor visual limitation rather than a functional bug.

### 2.10 Live Dashboard resets on server restart
**Issue:** The "Live Results" session-based tracking in the Evaluation Dashboard uses Streamlit's `st.session_state`, which does not persist across app restarts.
**Why unresolved:** This is expected, standard Streamlit behavior — `session_state` is explicitly scoped to a single running session. Persisting live results permanently would require adding a database or file-based storage layer, which was out of scope for this project's goals.
