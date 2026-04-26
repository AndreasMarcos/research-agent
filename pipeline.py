from pathlib import Path
import fitz
from models import Document
from rag.embedder import embed_documents
from rag.store import add_documents, collection_info
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── extração ───────────────────────────────────────────────────────────

def _extract_text(file_path: Path) -> list[Document]:
    """Extrai texto de um PDF página por página."""
    documents = []

    with fitz.open(file_path) as pdf:
        total_pages = len(pdf)
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            if _is_fragmented(text):
                print(f"[pipeline] Página {page_num} ignorada — texto fragmentado.")
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                }
            ))

    return documents


def _is_fragmented(text: str) -> bool:
    """Detecta páginas com texto fragmentado (figuras, imagens)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    short_lines = sum(1 for l in lines if len(l) <= 3)
    return short_lines / len(lines) > 0.5


# ── chunking ───────────────────────────────────────────────────────────

def _chunk_documents(docs: list[Document]) -> list[Document]:
    """Divide Documents em chunks menores com overlap recursivo."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    for doc in docs:
        doc_chunks = _recursive_split(
            doc.page_content,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            separators,
        )
        for i, chunk in enumerate(doc_chunks):
            if chunk.strip():
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                    }
                ))

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    parts = text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and len(separators) > 1:
                sub = _recursive_split(part, chunk_size, chunk_overlap, separators[1:])
                chunks.extend(sub[:-1])
                current = sub[-1] if sub else ""
            else:
                current = part

    if current:
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            result.append(tail + " " + chunks[i])
        return result

    return chunks


# ── ingestão pública ───────────────────────────────────────────────────

def ingest_pdf(file_path: str | Path) -> dict:
    """
    Processa e indexa um PDF no Qdrant.

    Args:
        file_path: caminho para o arquivo .pdf

    Returns:
        Dicionário com resumo do processamento.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Esperado arquivo .pdf, recebido: {path.suffix}")

    print(f"\n[pipeline] Ingerindo: {path.name}")

    docs = _extract_text(path)
    print(f"[pipeline] {len(docs)} página(s) extraída(s).")

    chunks = _chunk_documents(docs)
    print(f"[pipeline] {len(chunks)} chunk(s) gerado(s).")

    embedded = embed_documents(chunks)
    add_documents(embedded)

    return {
        "filename": path.name,
        "pages": len(docs),
        "chunks": len(chunks),
    }


def status() -> None:
    """Exibe informações sobre o índice atual."""
    info = collection_info()
    print(f"\n[pipeline] Collection: {info['name']} | Chunks: {info['count']}")