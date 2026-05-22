from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Global model (initialized once)
_model = None


def get_model():
    global _model

    if _model is None:
        print("[INFO] Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def generate_embeddings(chunks):
    model = get_model()

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts)

    print(f"[INFO] Generated {len(embeddings)} embeddings")

    return embeddings