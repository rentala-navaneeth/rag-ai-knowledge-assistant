import os
from config import DATA_PATH


def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if not file.endswith(".txt"):
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        documents.append({
            "text": text,
            "source": file
        })

    print(f"[INFO] Loaded {len(documents)} documents")
    return documents