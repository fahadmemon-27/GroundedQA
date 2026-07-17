from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load our chunks from file (1 chunk per line)
with open("data/processed/chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines()]

print(f"Loaded {len(chunks)} chunks")

# Create embeddings for all chunks
print("Creating embeddings... (this may take a minute)")
embeddings = model.encode(chunks, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

# Build FAISS index
dimension = embeddings.shape[1]  # should be 384
index = faiss.IndexFlatL2(dimension)  # simple, exact search index
index.add(np.array(embeddings).astype("float32"))

print(f"FAISS index built with {index.ntotal} vectors")

# Save the FAISS index to disk
faiss.write_index(index, "data/processed/faiss_index.bin")

# Save the chunks list too (we need to map index results back to actual text)
with open("data/processed/chunks_list.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Saved FAISS index and chunks list to data/processed/")