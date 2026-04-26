from sentence_transformers import SentenceTransformer
from models import Document
from config import EMBEDDING_MODEL


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Carregando modelo: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_documents(docs: list[Document]) -> list[tuple[Document, list[float]]]:
    if not docs:
        return []
    model = _get_model()
    texts = [doc.page_content for doc in docs]
    print(f"[embedder] Gerando embeddings para {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)
    return [(doc, vector.tolist()) for doc, vector in zip(docs, vectors)]


def embed_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Query não pode ser vazia.")
    model = _get_model()
    return model.encode(query).tolist()