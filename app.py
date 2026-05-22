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

# =========================
# CLEAN LOGS
# =========================
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
# UI HEADER
# =========================
st.title("🧠 AI Knowledge Assistant (RAG)")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-K Retrieval", 1, 5, 3)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.3)

# =========================
# CHAT MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# USER INPUT
# =========================
query = st.chat_input("Ask something...")

if query:
    logger.info(f"User query: {query}")

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # =========================
    # RETRIEVAL
    # =========================
    query_embedding = retriever.model.encode(query)
    results = retriever.store.search(query_embedding, top_k=top_k)

    # =========================
    # CONTEXT BUILDING
    # =========================
    context = ""
    for i, r in enumerate(results):
        context += f"[Source {i+1}]\n{r['text']}\n\n"

    logger.info(f"Context length: {len(context)}")

    # =========================
    # LLM CALL
    # =========================
    try:
        response = llm.generate(query, context, temperature)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        response = {"answer": "LLM error", "confidence": "low"}

    answer = response.get("answer", "Error")
    confidence = response.get("confidence", "low")

    # =========================
    # SAVE RESPONSE
    # =========================
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "sources": results,
        "context": context
    })

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

            # Confidence
            st.markdown(f"**Confidence:** `{msg.get('confidence', 'low')}`")

            # Sources
            st.markdown("**Sources:**")
            for i, src in enumerate(msg.get("sources", [])):
                st.markdown(f"- 📄 **Source {i+1}:** `{src['source']}`")

            # Expandable Context (WOW FEATURE)
            with st.expander("🔍 View Retrieved Context"):
                st.write(msg.get("context", ""))

# =========================
# WARNING
# =========================
if query and "I don't know" in answer:
    st.warning("Model could not find answer in context.")