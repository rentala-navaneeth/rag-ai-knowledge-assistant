import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(chunks)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        seen = set()
        results = []

        for i in I[0]:
            if i >= len(self.metadata):
                continue

            text = self.metadata[i]["text"]

            if text not in seen:
                results.append(self.metadata[i])
                seen.add(text)

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")

        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        print("[INFO] FAISS index saved")

    def load(self, path):
        self.index = faiss.read_index(path + ".index")

        with open(path + ".pkl", "rb") as f:
            self.metadata = pickle.load(f)

        print("[INFO] FAISS index loaded")