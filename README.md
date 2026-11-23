GroundedQA: Reducing LLM Hallucinations with RAG and RAD
A local, CPU-based question-answering system that compares three ways of generating answers from a knowledge base: a plain LLM baseline, Retrieval-Augmented Generation (RAG), and Retrieval-Augmented Decoding (RAD).

The project explores whether influencing token generation with retrieved evidence can improve answer groundedness beyond conventional prompt-based RAG. It also evaluates each answer with a Natural Language Inference (NLI) model and presents the comparison in an interactive Streamlit dashboard.
Project Overview
Large Language Models can produce fluent answers that are not supported by reliable evidence. GroundedQA compares three approaches under the same underlying language model:
Approach	How it works
LLM Baseline	Generates an answer from the model's learned knowledge, without retrieved evidence.
RAG	Retrieves relevant text chunks and places them in the prompt before normal generation.
RAD	Compares next-token predictions with and without evidence, then adjusts the token scores to give more weight to evidence-influenced choices.


A separate NLI-based evaluator checks how well each answer is supported by the retrieved evidence and calculates a groundedness score.
System Architecture
                         User Question
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
       LLM Baseline            RAG               RAD
       No retrieval      Evidence in prompt   Evidence-aware
                                               token decoding
             |                 |                 |
             +-----------------+-----------------+
                               |
                               v
                    NLI Groundedness Scorer
                               |
                               v
                      Streamlit Interface
                 Answers, scores and dashboard


SQuAD Dataset
      |
      v
Context Deduplication and Chunking
      |
      v
MiniLM Sentence Embeddings
      |
      v
FAISS Vector Index
      |
      v
Top-k Evidence Retrieval for RAG and RAD
How the Approaches Work
1. LLM Baseline
The language model answers the question without retrieval. This provides a reference point for comparing how much evidence-based methods improve groundedness.
2. Retrieval-Augmented Generation (RAG)
1. The question is converted into an embedding.
2. FAISS retrieves the most relevant text chunks.
3. The retrieved evidence is added to the model prompt.
4. The model is instructed to answer using the supplied context and to say when the answer is not available in it.
Generation uses the standard Hugging Face model.generate() method.
3. Retrieval-Augmented Decoding (RAD)
RAD uses the same retrieved evidence but goes beyond adding it to the prompt. At each generation step, it compares the model's next-token scores for two contexts: one containing the evidence and one without it.

The adjustment used by the implementation is:
adjusted_logits =
    logits_with_evidence
    + alpha * (logits_with_evidence - logits_without_evidence)
This increases the influence of tokens whose likelihood is strengthened by the evidence. RAD uses a custom token-generation loop, KV caching to avoid repeatedly processing the full sequence, and a repetition penalty to reduce repeated phrases.
Knowledge Base and Models
- Dataset: Stanford Question Answering Dataset (SQuAD), loaded through Hugging Face datasets.
- Dataset sample: The first 2,000 examples are used. Their contexts are deduplicated into 266 unique paragraphs and split into 759 chunks.
- Chunking: 400 characters per chunk with 50 characters of overlap.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings).
- Vector search: FAISS using IndexFlatL2; the default retrieval setting is top 3 chunks.
- Generation model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, shared across all three generation approaches for a controlled comparison.
- Groundedness evaluator: cross-encoder/nli-deberta-v3-small.
The project was designed to run locally on CPU and does not use paid LLM APIs.
Groundedness Evaluation
The NLI scorer splits each answer into sentences and compares each sentence with the retrieved evidence. It classifies the relationship as:
- Supported (entailment): full credit.
- Unclear (neutral): half credit.
- Contradicted: no credit.
The sentence-level results are averaged into a groundedness score from 0% to 100%.

A small hand-labelled validation set of 10 evidence/claim pairs was used to sanity-check the scorer, with 90% agreement with the human labels on that sample.
Evaluation Results
The documented evaluation run reported the following average groundedness scores:
Pipeline	Average groundedness
LLM Baseline	41.7%
RAG	50.0%
RAD	58.3%


These results show the observed trend Baseline < RAG < RAD in the evaluated sample. They are indicative rather than statistically conclusive: the final evaluation used a small sample of approximately 6–10 questions because RAD is computationally expensive on CPU. RAD also did not outperform RAG on every individual question.
Streamlit Application
The application provides two tabs:
- Home: Enter a question and compare the LLM Baseline, RAG, and RAD answers side by side, each with its groundedness score.
- Evaluation Dashboard: View the saved benchmark averages and live averages for questions asked during the current session, with a comparison chart.
The interface uses a dark, glassmorphism-inspired theme. Live session results are held in Streamlit session state and reset when the app session/server is restarted.
Tech Stack
Area	Tools
Language	Python 3.11.9
Deep learning	PyTorch (CPU build)
Language model and token generation	Hugging Face Transformers, TinyLlama
Text embeddings	Sentence Transformers, all-MiniLM-L6-v2
Vector retrieval	FAISS CPU
Dataset	Hugging Face Datasets, SQuAD
NLI evaluation	Sentence Transformers CrossEncoder, DeBERTa NLI
Web interface	Streamlit
Charts	Altair
Data handling	NumPy, pandas
Model loading support	Accelerate


Repository Structure
GroundedQA/
├── app.py
├── build_index.py
├── chunk_data.py
├── evaluate.py
├── evaluation_results.json
├── llm_baseline.py
├── nli_evaluation.py
├── nli_scorer.py
├── rad_pipeline.py
├── rag_pipeline.py
├── requirements.txt
├── PROJECT_OVERVIEW.md
├── TECH_STACK.md
├── LIMITATIONS.md
├── data/
│   └── processed/
└── temporary_data_info/
Main files
File	Responsibility
app.py	Streamlit application and evaluation dashboard.
chunk_data.py	Loads SQuAD, deduplicates contexts and creates text chunks.
build_index.py	Generates embeddings and builds the FAISS index.
llm_baseline.py	Implements the no-retrieval baseline.
rag_pipeline.py	Implements evidence retrieval and prompt-based generation.
rad_pipeline.py	Implements evidence-aware, token-level decoding.
nli_scorer.py	Scores answer sentences against evidence using NLI.
nli_evaluation.py	Checks the NLI scorer against hand-labelled examples.
evaluate.py	Runs the comparative evaluation and saves results.
evaluation_results.json	Stores the documented benchmark results.
PROJECT_OVERVIEW.md	Detailed project motivation, methodology, results and architecture.
TECH_STACK.md	Explains the tools, libraries and models used.
LIMITATIONS.md	Records resolved issues and known limitations.


Setup and Installation
Prerequisites
- Python 3.11.9 (the project was developed and tested with this version).
- Git.
- Internet access for installing dependencies and downloading the dataset/models.
- Sufficient RAM and disk space for local model files.
1. Clone the repository
git clone https://github.com/fahadmemon-27/GroundedQA.git
cd GroundedQA
2. Create and activate a virtual environment
Windows CMD:
py -3.11 -m venv venv
venv\Scripts\activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Prepare the data and retrieval index
Run the preparation scripts in this order:
python chunk_data.py
python build_index.py
These steps prepare the SQuAD-derived text chunks, embeddings and FAISS index required by the retrieval pipelines.
5. Launch the application
python -m streamlit run app.py
Streamlit will print a local URL in the terminal. Open that URL in your browser.
Note: The first run may take time because the dataset and model files need to be downloaded and loaded locally. The exact dependency versions are listed in requirements.txt.

Run the Evaluation
To run the comparative evaluation script:
python evaluate.py
The script evaluates the three approaches and writes its results to evaluation_results.json.

To validate the NLI scorer using the hand-labelled examples:
python nli_evaluation.py
Limitations and Considerations
This is an experimental research project, not a guarantee that generated answers are factually correct.
- The evaluation sample is small (approximately 6–10 questions), so the reported averages are not statistically definitive.
- RAD is slower than baseline generation and RAG because it performs two model forward passes per generated token. KV caching reduces redundant computation but does not remove this cost.
- RAD does not outperform RAG on every question.
- The NLI scorer can miss false-equivalence claims and can be sensitive to phrasing.
- Sentence-level scoring can penalize a long sentence for one incorrect detail.
- The groundedness score is a directional signal of evidence support, not a definitive measure of factual truth.
- Live dashboard results are session-based and are not persisted across server restarts.
- The application runs local models on CPU, so response time and evaluation scale are constrained by available hardware.
For the full record of resolved and unresolved issues, see [`LIMITATIONS.md`](LIMITATIONS.md).
Future Work
Potential next steps documented for the project include:
- Expand evaluation to a larger set of questions.
- Explore an LLM-based evaluator as a complement to the NLI scorer.
- Investigate clause-level scoring for long, multi-claim sentences.
- Test larger generation models when more compute is available.
- Persist live evaluation results between sessions.
Author
Fahad Memon

GitHub: @fahadmemon-27

Repository: GroundedQA
Built as a comparative study of evidence-grounded question answering, retrieval, and token-level decoding.
