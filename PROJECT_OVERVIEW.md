# Hallucination Reduction in LLM-based QA using RAG and RAD

## A Comparative Study of LLM, Retrieval-Augmented Generation, and Retrieval-Augmented Decoding

---

## 1. Project Overview

Large Language Models (LLMs) are prone to **hallucination** — generating fluent, confident-sounding answers that are factually incorrect or entirely fabricated. This project builds and compares three approaches to answering questions, in increasing order of how tightly they are grounded in real evidence, and measures how much each approach actually reduces hallucination.

**The three approaches compared:**

1. **LLM Baseline** — the model answers purely from its own trained knowledge, with no external information. This represents the highest-risk, most hallucination-prone approach, and serves as the control/reference point.

2. **RAG (Retrieval-Augmented Generation)** — before answering, relevant evidence is retrieved from a knowledge base and inserted into the model's prompt. The model is instructed to answer using only that evidence. This is the industry-standard approach to reducing hallucination.

3. **RAD (Retrieval-Augmented Decoding)** — the project's main contribution. Rather than simply *suggesting* evidence in the prompt (as RAG does) and hoping the model follows it, RAD actively influences the model's word-by-word generation. At every single generated word, the model's prediction is computed twice — once *with* the evidence, once *without* — and the difference is used to boost words that only became likely because of the evidence, while suppressing words the model would have guessed anyway from memory. This makes the model's output more mechanically tied to the retrieved evidence, rather than merely encouraged to use it.

**A fourth component — the hallucination detector** — automatically scores how well each generated answer is actually supported by the evidence it was given, using a Natural Language Inference (NLI) model. This produces a **groundedness score** (0–100%) for every answer, allowing the three approaches to be compared numerically, not just by eye.

All of this is wrapped in a live web interface where a user can type any question and see all three answers side by side, along with their groundedness scores.

---

## 2. Motivation

Hallucination is one of the most significant open problems limiting trust in LLM-based systems, especially in domains where factual accuracy matters (research, healthcare-adjacent information, education, journalism support). Retrieval-Augmented Generation is a widely used mitigation, but it has a known weakness: **the model is only encouraged, not forced, to use the retrieved evidence** — it can still ignore it and hallucinate anyway.

This project explores whether a more direct intervention — constraining the generation process itself, token by token, based on evidence — produces measurably more grounded answers than prompt-based RAG alone, while remaining achievable using entirely free, local, open-source tools (no paid APIs), running on a standard CPU-only machine.

---

## 3. System Architecture

```
                              ┌─────────────────┐
                              │   User Question   │
                              └────────┬─────────┘
                                       │
                       ┌───────────────┼───────────────┐
                       │               │               │
                       ▼               ▼               ▼
              ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
              │ LLM Baseline│  │     RAG      │  │     RAD      │
              │ (no evidence)│  │ (evidence in │  │ (evidence    │
              │             │  │   prompt)    │  │  constrains  │
              │             │  │              │  │  each token) │
              └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
                     │                │                  │
                     └────────────────┼──────────────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   NLI Groundedness      │
                          │   Scorer (per sentence)  │
                          └────────────┬────────────┘
                                       ▼
                          ┌────────────────────────┐
                          │  Streamlit Web UI        │
                          │  (answers + scores +     │
                          │   evaluation dashboard)  │
                          └────────────────────────┘
```

**Shared retrieval layer (used by both RAG and RAD):**
```
SQuAD dataset → deduplicated paragraphs → chunked text →
MiniLM embeddings → FAISS vector index → top-k retrieval per question
```

---

## 4. Dataset

- **Source:** `rajpurkar/squad` (Stanford Question Answering Dataset), via HuggingFace `datasets`.
- **Why this dataset:** it provides clean, separate `question`, `context` (evidence paragraph), and `answers` (ground truth) fields — ideal for a retrieval-based QA system. An earlier dataset choice lacked a proper separate context field, which caused early bugs and was replaced with SQuAD.
- **Scale used:** first 2,000 examples, deduplicated down to 266 unique paragraphs (since many questions share the same source paragraph), chunked into 759 retrieval chunks (400 characters per chunk, 50-character overlap to avoid cutting sentences at chunk boundaries).

---

## 5. Methodology

### 5.1 Retrieval
Text chunks are embedded using `all-MiniLM-L6-v2` (384-dimensional sentence embeddings) and indexed with FAISS (`IndexFlatL2`) for exact nearest-neighbor search. Given a question, its embedding is compared against all chunk embeddings, and the top-k (default 3) most relevant chunks are retrieved as evidence.

### 5.2 Generation model
All three pipelines use the same underlying LLM — **TinyLlama-1.1B-Chat** — to ensure a fair, controlled comparison (identical model weights, only the generation *method* differs between pipelines). TinyLlama was chosen specifically because:
- It is small enough to run at reasonable speed on CPU-only hardware.
- It is coherent enough to produce meaningful, comparably fluent answers.
- Unlike black-box tools (e.g. Ollama), it can be loaded directly via the `transformers` library, exposing the token-level internals required for RAD — meaning no model swap was needed when moving from baseline/RAG to RAD.

### 5.3 RAG implementation
Retrieved evidence chunks are concatenated and inserted into the prompt with an explicit instruction: *"Answer the question using ONLY the information in the context below. If the answer is not in the context, say 'I don't know'."* Generation then proceeds normally via `model.generate()`.

### 5.4 RAD implementation
Because RAD requires comparing two predictions (with-evidence vs without-evidence) at every generation step, it cannot use the standard `model.generate()` function. Instead, a custom generation loop was built:

1. Two prompts are prepared — one with evidence, one without.
2. At each step, the model is run once on each prompt to get next-word probability scores (logits).
3. The adjustment formula is applied:
   `adjusted_logits = logits_with_evidence + α × (logits_with_evidence − logits_without_evidence)`
   This boosts words that became more likely *specifically because of* the evidence, and suppresses words the model would have picked from memory regardless.
4. The highest-scoring word after adjustment is selected and fed back into both sequences for the next step.
5. This repeats until an end-of-sequence token is produced or the token limit is reached.

**Performance optimizations applied:**
- **KV caching** — avoids reprocessing the entire growing sequence at every step; only the newest token is processed, dramatically reducing redundant computation.
- **Repetition penalty** — discourages the model from repeating already-used words, fixing an observed degeneration issue (RAD initially sometimes repeated phrases verbatim).

### 5.5 Hallucination scoring (NLI)
Each generated answer is split into individual sentences. Each sentence is compared against the retrieved evidence using `cross-encoder/nli-deberta-v3-small`, a Natural Language Inference model that classifies the relationship between an evidence passage (premise) and a claim (hypothesis) as:
- **Supported** (entailment) — full credit
- **Unclear** (neutral) — half credit
- **Contradicted** — no credit

The average across all sentences in an answer produces the final **groundedness score** for that answer.

A small hand-labeled validation set (10 evidence/claim pairs with human-assigned correct labels) was used to sanity-check the scorer itself, achieving **90% agreement** with human judgment — with its one error matching an already-identified, documented weakness (see Limitations).

---

## 6. Results

Evaluated across a diverse, evidence-matched sample of questions:

| Pipeline | Average Groundedness Score |
|---|---|
| LLM Baseline | 41.7% |
| RAG | 50.0% |
| RAD | 58.3% |

This trend — **Baseline < RAG < RAD** — supports the project's core hypothesis: retrieval improves grounding over an ungrounded baseline, and constraining decoding at the token level improves grounding further still, on average.

**Important caveat:** individual question results varied more than the averages suggest (RAD outperformed RAG substantially on some questions, and underperformed on others), and the sample size (6–10 questions) is small due to CPU-only compute constraints. The aggregate trend is a meaningful, real signal, but not a statistically definitive result — see Limitations for full discussion.

---

## 7. User Interface

Built with **Streamlit**, featuring:
- A **Home tab** — single question input, triggers all three pipelines simultaneously, displays each answer alongside a color-coded groundedness badge (green/yellow/red).
- An **Evaluation Dashboard tab** — shows both a fixed benchmark reference (from a saved evaluation run) and a live, session-based running average that updates dynamically as more questions are asked, with a color-coded comparison chart.
- A dark, glassmorphism-inspired visual theme (frosted-glass panels, blurred backgrounds).
- Enter-to-submit support alongside the Submit button.

---

## 8. Key Findings

1. **Retrieval alone measurably reduces hallucination** — RAG's average groundedness (50.0%) was noticeably higher than the ungrounded baseline (41.7%).
2. **Constraining decoding further improves grounding, on average** — RAD's average (58.3%) exceeded RAG's, supporting the core hypothesis that token-level evidence-weighting adds value beyond prompt-based grounding alone.
3. **RAD is not universally superior on a per-question basis** — it occasionally underperformed RAG when the underlying LLM's own knowledge was confused (e.g., conflating "antibiotics" with "anti-inflammatory drugs"), and it tends to produce longer, more detailed answers that our scoring method is less generous toward than RAG's shorter, safer phrasing.
4. **The choice of evaluator matters as much as the choice of generator** — several of the "surprising" or seemingly inconsistent results traced back not to the generation pipelines, but to known, well-understood limitations of the small NLI model used for scoring (strict phrasing sensitivity, inability to catch false-equivalence claims, coarseness on long sentences). This distinction — generation quality vs. evaluation quality — was one of the most important lessons of the project.

A full, itemized log of every limitation encountered, what was fixed and how, and what remains open with reasoning, is maintained separately in `LIMITATIONS.md`.

---

## 9. Constraints and Design Principles

- **No paid APIs** — no OpenAI, Gemini, or other paid LLM services used anywhere in the pipeline.
- **Fully local** — all models (embedding, generation, NLI) run locally.
- **CPU-only** — developed and tested with no GPU access, which directly shaped several design decisions (model size choices, KV caching necessity, evaluation sample size).
- **Open-source only** — every model used (TinyLlama, MiniLM, DeBERTa-based NLI model) is freely available and open-source.

---

## 10. Future Work

- Scale evaluation to a larger question sample (50+) given more compute/time.
- Explore a constrained LLM-based judge as an alternative or complement to the small NLI scorer, to better handle longer, well-formed, paraphrased answers.
- Investigate clause-level (not just sentence-level) scoring to reduce the impact of single incorrect details dragging down otherwise-correct long sentences.
- Test with larger, non-CPU-constrained generation models to see whether RAD's advantage over RAG grows, shrinks, or changes character with a stronger base model.
- Persist "live" evaluation results across sessions (currently reset on app restart) via lightweight file or database storage.

---

## 11. Repository Structure (Key Files)

| File | Purpose |
|---|---|
| `chunk_data.py` | Downloads SQuAD, deduplicates, and chunks text into the retrieval knowledge base |
| `build_index.py` | Builds embeddings and the FAISS retrieval index |
| `llm_baseline.py` | LLM-only baseline pipeline |
| `rag_pipeline.py` | RAG pipeline (retrieval + prompt injection) |
| `rad_pipeline.py` | RAD pipeline (retrieval + token-level constrained decoding) |
| `nli_scorer.py` | Sentence-level NLI-based groundedness scorer |
| `nli_evaluation.py` | Hand-labeled validation of the NLI scorer's own accuracy |
| `evaluate.py` | Runs all three pipelines across multiple questions, scores results, saves `evaluation_results.json` |
| `app.py` | Streamlit web interface |
| `LIMITATIONS.md` | Full log of resolved and unresolved project limitations |
| `TECH_STACK.md` | Ordered summary of every tool/library used and why |

---

*This project was built iteratively and tested at every stage — from raw text chunking through to a live, interactive web interface — with particular attention paid to honestly identifying and documenting the real limitations of each design choice, rather than only presenting favorable results.*
