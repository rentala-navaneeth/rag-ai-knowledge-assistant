from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

emb = model.encode("test sentence")

print("✅ Embedding size:", len(emb))