from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import generate_embeddings
from src.vector_store import VectorStore
from src.retrieve import Retriever
from src.llm_client import LLMClient

# Build RAG pipeline
docs = load_documents()
chunks = chunk_documents(docs)
embeddings = generate_embeddings(chunks)

dim = len(embeddings[0])
store = VectorStore(dim)
store.add(embeddings, chunks)

retriever = Retriever(store)
llm = LLMClient()

# Query
query = "What are transformers?"

# Retrieve
results = retriever.retrieve(query)

# Build context
context = "\n\n".join([r["text"] for r in results])

print("\n[Context]\n", context)

# Generate answer
response = llm.generate(query, context)

print("\n[Answer]\n", response["answer"])
print("\n[Confidence]\n", response["confidence"])