from src.embed import get_model
from config import TOP_K


class Retriever:
    def __init__(self, store):
        self.model = get_model()
        self.store = store

    def retrieve(self, query):
        query_emb = self.model.encode(query)
        results = self.store.search(query_emb, top_k=TOP_K)
        return results