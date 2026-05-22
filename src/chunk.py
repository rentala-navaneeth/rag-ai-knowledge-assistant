from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(docs):
    results = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            results.append({
                "text": chunk,
                "source": doc["source"]
            })

    print(f"[INFO] Created {len(results)} chunks")
    return results