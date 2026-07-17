from datasets import load_dataset
import os

MAX_DOCS = 2000
CHUNK_SIZE = 400
OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Load dataset
dataset = load_dataset("rajpurkar/squad")

# Take first MAX_DOCS examples
sample = dataset["train"].select(range(MAX_DOCS))

# Deduplicate contexts (many questions share the same paragraph)
unique_contexts = list(set(sample["context"]))

print(f"Total examples: {MAX_DOCS}")
print(f"Unique contexts after deduplication: {len(unique_contexts)}")
print()

# Chunk every unique context
all_chunks = []
for context in unique_contexts:
    chunks = chunk_text(context)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

# Save to disk
os.makedirs("data/processed", exist_ok=True)

with open("data/processed/chunks.txt", "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        # Replace newlines within a chunk so each chunk stays on ONE line in the file
        clean_chunk = chunk.replace("\n", " ").strip()
        f.write(clean_chunk + "\n")

print("Saved chunks to data/processed/chunks.txt")