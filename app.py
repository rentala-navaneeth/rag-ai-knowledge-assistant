import streamlit as st
import os
import logging

from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import generate_embeddings
from src.vector_store import VectorStore
from src.retrieve import Retriever
from src.llm_client import LLMClient
from src.logger import logger
from config import FAISS_INDEX_PATH

# Clean transformer logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="AI RAG Assistant", layout="wide")


# =========================
# LOAD SYSTEM (CACHED)
# =========================
@st.cache_resource
def load_system():
    logger.info("Initializing system...")

    store = VectorStore(384)

    # Load or build FAISS
    if os.path.exists(FAISS_INDEX_PATH + ".index"):
        store.load(FAISS_INDEX_PATH)
    else:
        docs = load_documents()
        chunks = chunk_documents(docs)
        embeddings = generate_embeddings(chunks)

        store.add(embeddings, chunks)
        store.save(FAISS_INDEX_PATH)

    retriever = Retriever(store)
    llm = LLMClient()

    return retriever, llm


retriever, llm = load_system()

# =========================
# UI
# =========================
st.title("🧠 AI Knowledge Assistant (RAG)")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask something...")

if query:
    logger.info(f"User query: {query}")

    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieval
    results = retriever.retrieve(query)

    # Structured context
    context = ""
    for i, r in enumerate(results):
        context += f"[Source {i+1}]\n{r['text']}\n\n"

    logger.info(f"Context length: {len(context)}")

    # LLM
    response = llm.generate(query, context)

    answer = response.get("answer", "Error")
    confidence = response.get("confidence", "low")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "sources": results
    })

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

        st.markdown(f"**Confidence:** {msg.get('confidence', 'low')}")

        st.markdown("**Sources:**")
        for i, src in enumerate(msg.get("sources", [])):
            st.write(f"Source {i+1}: {src['source']}")

# =========================
# NO ANSWER WARNING
# =========================
if query and "I don't know" in answer:
    st.warning("Model could not find answer in context.")