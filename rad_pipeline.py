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
model.eval()
print("All components ready.\n")


def retrieve_chunks(question, top_k=3):
    question_embedding = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved = [chunks[idx] for idx in indices[0]]
    return retrieved


def generate_rad_answer(question, top_k=3, max_new_tokens=100, alpha=1.0, repetition_penalty=1.3):
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    evidence = "\n".join(retrieved_chunks)

    messages_with = [{"role": "user", "content": f"Context: {evidence}\n\nQuestion: {question}"}]
    messages_without = [{"role": "user", "content": f"Question: {question}"}]

    prompt_with = tokenizer.apply_chat_template(messages_with, tokenize=False, add_generation_prompt=True)
    prompt_without = tokenizer.apply_chat_template(messages_without, tokenize=False, add_generation_prompt=True)

    tokenizer.pad_token = tokenizer.eos_token
    inputs_with = tokenizer(prompt_with, return_tensors="pt")
    inputs_without = tokenizer(prompt_without, return_tensors="pt")

    generated_ids_with = inputs_with["input_ids"]
    generated_ids_without = inputs_without["input_ids"]

    past_with = None
    past_without = None

    generated_tokens = []
    next_input_with = generated_ids_with
    next_input_without = generated_ids_without

    for step in range(max_new_tokens):
        with torch.no_grad():
            output_with = model(next_input_with, past_key_values=past_with, use_cache=True)
            output_without = model(next_input_without, past_key_values=past_without, use_cache=True)

        past_with = output_with.past_key_values
        past_without = output_without.past_key_values

        logits_with = output_with.logits[0, -1, :]
        logits_without = output_without.logits[0, -1, :]

        adjusted_logits = logits_with + alpha * (logits_with - logits_without)

        # Repetition penalty: discourage picking tokens already used in this answer
        for used_token in set(generated_tokens):
            if adjusted_logits[used_token] > 0:
                adjusted_logits[used_token] /= repetition_penalty
            else:
                adjusted_logits[used_token] *= repetition_penalty

        next_token_id = torch.argmax(adjusted_logits).unsqueeze(0).unsqueeze(0)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id.item())

        next_input_with = next_token_id
        next_input_without = next_token_id

    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer, retrieved_chunks


if __name__ == "__main__":
    question = "What is on top of the Main Building at Notre Dame?"
    answer, evidence_used = generate_rad_answer(question)

    print(f"Question: {question}\n")
    print("Retrieved evidence used:")
    for i, chunk in enumerate(evidence_used):
        print(f"[{i+1}] {chunk[:150]}...")
    print()
    print(f"RAD Answer: {answer}")