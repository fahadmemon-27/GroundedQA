from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import pickle
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index and chunks...")
index = faiss.read_index("data/processed/faiss_index.bin")
with open("data/processed/chunks_list.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Loading TinyLlama model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
print("All components ready.\n")


def retrieve_chunks(question, top_k=3):
    """Same retrieval logic we already tested in test_retrieval.py"""
    question_embedding = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved = [chunks[idx] for idx in indices[0]]
    return retrieved


def generate_rag_answer(question, top_k=3, max_new_tokens=100):
    """
    RAG: retrieve relevant chunks, inject into prompt, then generate.
    """
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)

    # Combine retrieved chunks into one evidence block
    evidence = "\n".join(retrieved_chunks)

    # Build a prompt that includes the evidence + instructs the model to use it
    user_message = (
        f"Answer the question using ONLY the information in the context below. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{evidence}\n\n"
        f"Question: {question}"
    )

    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full_text.split("<|assistant|>")[-1].strip()

    return answer, retrieved_chunks


# Quick test if run directly
if __name__ == "__main__":
    question = "What is the Grotto at Notre Dame?"

    answer, evidence_used = generate_rag_answer(question)

    print(f"Question: {question}\n")
    print("Retrieved evidence used:")
    for i, chunk in enumerate(evidence_used):
        print(f"[{i+1}] {chunk[:150]}...")
    print()
    print(f"RAG Answer: {answer}")
