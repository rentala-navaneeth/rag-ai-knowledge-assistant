# 🧠 AI Knowledge Assistant (Hybrid RAG System)

## 🔹 Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** that combines semantic search with large language model (LLM) inference to generate **grounded, context-aware responses**.

The system retrieves relevant documents using FAISS-based vector search and generates answers using a **GPU-hosted Mistral LLM** via a remote API, improving factual accuracy and reducing hallucinations.

---

## 🏗 Architecture

User Query
↓
Retriever (FAISS + Embeddings)
↓
Context Builder
↓
LLM API (Mistral - Colab)
↓
Response Parsing + Confidence
↓
Streamlit Chat UI

---

## ⚙️ Features

- 🔍 Semantic search using FAISS
- 🧠 Embedding generation with Sentence Transformers
- 🤖 LLM inference using Mistral (GPU via Colab)
- 🔗 Retrieval-Augmented Generation (RAG)
- 📦 Persistent FAISS index (no recomputation)
- 💬 Chat-style UI using Streamlit
- 📊 Confidence scoring for responses
- 🧾 Source attribution for transparency
- 📈 Evaluation pipeline (accuracy + latency)
- 🪵 Logging system for debugging and monitoring

---

## 🛠 Tech Stack

- Python
- FAISS (Vector Search)
- Sentence Transformers
- Flask (LLM API)
- Streamlit (Frontend UI)
- Google Colab (GPU inference)
- ngrok (API exposure)

---

## 🚀 How to Run

### 1. Start LLM API (Colab)
- Run the Flask + Mistral notebook
- Copy the ngrok public URL

### 2. Update Config

```python
LLM_API_URL = "https://unbounded-refund-luckless.ngrok-free.dev/generate"

3. Run Application
streamlit run app.py
4. Run Evaluation
python eval.py
📌 Example Output
Q: What is RAG?

A: Retrieval-Augmented Generation (RAG) combines information retrieval with text generation...

Confidence: high

Sources:
- rag.txt
- faiss.txt
- vector_db.txt
📊 Metrics

The system is evaluated using:

Retrieval Accuracy → Measures if correct source documents are retrieved
Response Latency → Time taken for LLM response generation
Context Size → Amount of information passed to LLM
Grounding Check → Ensures responses are derived from retrieved context
🧠 Key Design Decisions
Modular architecture separating retrieval, embedding, and generation
Persistent FAISS index to eliminate recomputation overhead
Structured JSON output to enforce response consistency
Context grounding to reduce hallucinations
Lightweight API-based LLM integration (Colab + ngrok)

This project demonstrates:

End-to-end AI system design
Integration of retrieval + generation pipelines
Real-world LLM deployment architecture
Performance-aware engineering (latency, caching, persistence)