from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import generate_embeddings
from src.vector_store import VectorStore
from src.retrieve import Retriever

# Build pipeline
docs = load_documents()
chunks = chunk_documents(docs)
embeddings = generate_embeddings(chunks)

# Create store
dim = len(embeddings[0])
store = VectorStore(dim)
store.add(embeddings, chunks)

# Create retriever
retriever = Retriever(store)

# Query
query = "Explain transformers"
results = retriever.retrieve(query)

print("\nRetrieved:\n")

for r in results:
    print(r["text"])
    print("-" * 50)