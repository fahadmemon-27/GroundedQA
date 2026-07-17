from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama model and tokenizer (one-time load)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
print("Model ready.\n")

def generate_baseline_answer(question, max_new_tokens=100):
    """
    Pure LLM baseline: no retrieval, no evidence.
    Just question -> model -> answer.
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the assistant's reply (after the last "assistant" turn)
    answer = full_text.split("<|assistant|>")[-1].strip()
    return answer


# Quick test if this file is run directly
if __name__ == "__main__":
    test_questions = [
        "What is on top of the Main Building at Notre Dame?",
        "What is the Grotto at Notre Dame?"
    ]

    for q in test_questions:
        print(f"Question: {q}")
        answer = generate_baseline_answer(q)
        print(f"Answer: {answer}")
        print("-" * 50)