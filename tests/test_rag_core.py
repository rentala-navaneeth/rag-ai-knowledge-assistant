from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import generate_embeddings
from src.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

# Load pipeline
docs = load_documents()
chunks = chunk_documents(docs)
embeddings = generate_embeddings(chunks)

# Build vector store
dim = len(embeddings[0])
store = VectorStore(dim)
store.add(embeddings, chunks)

# Query test
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "What are transformers?"
query_emb = model.encode(query)

results = store.search(query_emb)

print("\nTop Result:\n")
print(results[0]["text"])