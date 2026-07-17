from datasets import load_dataset
from llm_baseline import generate_baseline_answer
from rag_pipeline import generate_rag_answer, retrieve_chunks
from rad_pipeline import generate_rad_answer
from nli_scorer import score_answer
import json

# IMPORTANT: only pick questions from the SAME 2000 examples we used to
# build chunks.txt / the FAISS index (chunk_data.py used MAX_DOCS = 2000).
# Picking from the full dataset caused topic mismatches with retrieval.
print("Selecting diverse test questions...")
dataset = load_dataset("rajpurkar/squad")
sample = dataset["train"].select(range(2000))

seen_titles = set()
diverse_questions = []

for example in sample:
    if example["title"] not in seen_titles:
        seen_titles.add(example["title"])
        diverse_questions.append(example["question"])
    if len(diverse_questions) >= 10:
        break

TEST_QUESTIONS = diverse_questions
print("Selected questions:")
for q in TEST_QUESTIONS:
    print(" -", q)
print()

results = []

for question in TEST_QUESTIONS:
    print(f"\n=== Question: {question} ===")

    evidence_chunks = retrieve_chunks(question)

    print("Running LLM baseline...")
    baseline_answer = generate_baseline_answer(question)
    baseline_score = score_answer(baseline_answer, evidence_chunks)

    print("Running RAG...")
    rag_answer, _ = generate_rag_answer(question)
    rag_score = score_answer(rag_answer, evidence_chunks)

    print("Running RAD...")
    rad_answer, _ = generate_rad_answer(question)
    rad_score = score_answer(rad_answer, evidence_chunks)

    results.append({
        "question": question,
        "baseline": {"answer": baseline_answer, "groundedness": baseline_score["groundedness_score"]},
        "rag": {"answer": rag_answer, "groundedness": rag_score["groundedness_score"]},
        "rad": {"answer": rad_answer, "groundedness": rad_score["groundedness_score"]},
    })

print("\n\n=== SUMMARY ===")
print(f"{'Question':<50} {'Baseline':<10} {'RAG':<10} {'RAD':<10}")
for r in results:
    q_short = r["question"][:47] + "..." if len(r["question"]) > 47 else r["question"]
    print(f"{q_short:<50} {r['baseline']['groundedness']:<10.1f} {r['rag']['groundedness']:<10.1f} {r['rad']['groundedness']:<10.1f}")

avg_baseline = sum(r["baseline"]["groundedness"] for r in results) / len(results)
avg_rag = sum(r["rag"]["groundedness"] for r in results) / len(results)
avg_rad = sum(r["rad"]["groundedness"] for r in results) / len(results)

print(f"\n{'AVERAGE':<50} {avg_baseline:<10.1f} {avg_rag:<10.1f} {avg_rad:<10.1f}")

print("\n\n=== DETAILED ANSWERS (for inspection) ===")
for r in results:
    print(f"\nQuestion: {r['question']}")
    print(f"Baseline ({r['baseline']['groundedness']:.0f}%): {r['baseline']['answer']}")
    print(f"RAG ({r['rag']['groundedness']:.0f}%): {r['rag']['answer']}")
    print(f"RAD ({r['rad']['groundedness']:.0f}%): {r['rad']['answer']}")

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nFull results saved to evaluation_results.json")