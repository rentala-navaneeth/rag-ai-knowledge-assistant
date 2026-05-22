from src.vector_store import VectorStore
from src.retrieve import Retriever
from src.llm_client import LLMClient
from config import FAISS_INDEX_PATH

import time

# =========================
# LOAD SYSTEM
# =========================
store = VectorStore(384)
store.load(FAISS_INDEX_PATH)

retriever = Retriever(store)
llm = LLMClient()

# =========================
# TEST CASES
# =========================
test_cases = [
    {"query": "What is RAG?", "expected": "rag.txt"},
    {"query": "What is FAISS?", "expected": "faiss.txt"},
    {"query": "Explain embeddings", "expected": "embeddings.txt"},
    {"query": "What are transformers?", "expected": "transformers.txt"},
]

# =========================
# METRICS TRACKERS
# =========================
correct_retrievals = 0
total_latency = 0
grounded_count = 0

# =========================
# RUN EVALUATION
# =========================
for t in test_cases:
    query = t["query"]
    expected = t["expected"]

    print("\n" + "=" * 50)
    print("Query:", query)

    # 🔍 Retrieval
    results = retriever.retrieve(query)
    sources = [r["source"] for r in results]

    print("Retrieved Sources:", sources)

    # 📊 Retrieval Accuracy
    if expected in sources:
        correct_retrievals += 1
        print("Retrieval: ✅ Correct")
    else:
        print("Retrieval: ❌ Incorrect")

    # 🧠 Context
    context = "\n".join([r["text"] for r in results])
    print("Context Length:", len(context))

    # ⚡ Latency
    start = time.time()
    response = llm.generate(query, context)
    end = time.time()

    latency = end - start
    total_latency += latency

    print(f"Latency: {latency:.2f} sec")

    answer = response.get("answer", "")
    confidence = response.get("confidence", "low")

    print("Answer:", answer)
    print("Confidence:", confidence)

    # 🔍 Grounding Check
    if answer.lower() in context.lower():
        grounded_count += 1
        print("Grounding: ✅ Answer supported by context")
    else:
        print("Grounding: ⚠️ Possibly hallucinated")

# =========================
# FINAL METRICS
# =========================
total = len(test_cases)

retrieval_accuracy = correct_retrievals / total
avg_latency = total_latency / total
grounding_score = grounded_count / total

print("\n" + "=" * 50)
print("📊 FINAL METRICS")
print("=" * 50)

print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}")
print(f"Average Latency: {avg_latency:.2f} sec")
print(f"Grounding Score: {grounding_score:.2f}")