from src.vector_store import VectorStore
from src.retrieve import Retriever
from src.llm_client import LLMClient
from config import FAISS_INDEX_PATH

store = VectorStore(384)
store.load(FAISS_INDEX_PATH)

retriever = Retriever(store)
llm = LLMClient()

queries = [
    "What is RAG?",
    "Explain embeddings",
    "What is FAISS?",
    "What are transformers?"
]

for q in queries:
    results = retriever.retrieve(q)
    context = "\n".join([r["text"] for r in results])
    response = llm.generate(q, context)

    print("\nQ:", q)
    print("A:", response["answer"])
    print("Confidence:", response["confidence"])