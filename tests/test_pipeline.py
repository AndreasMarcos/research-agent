import pytest

import pipeline
from models import Document


# ── _is_fragmented ─────────────────────────────────────────────────────

def test_is_fragmented_detecta_linhas_curtas():
    text = "\n".join(["a", "bc", "1", "x", "yz", "linha normal de texto"])
    assert pipeline._is_fragmented(text) is True


def test_is_fragmented_texto_normal():
    text = "\n".join(f"Esta é a linha {i} de um parágrafo comum." for i in range(10))
    assert pipeline._is_fragmented(text) is False


def test_is_fragmented_poucas_linhas_nunca_fragmentado():
    # menos de 5 linhas não-vazias → nunca considerado fragmentado
    assert pipeline._is_fragmented("a\nb\n\nc\nd") is False


# ── _recursive_split ───────────────────────────────────────────────────

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def test_recursive_split_texto_curto_retorna_um_chunk():
    assert pipeline._recursive_split("texto curto", 100, 10, SEPARATORS) == ["texto curto"]


def test_recursive_split_respeita_tamanho_sem_overlap():
    text = " ".join(f"palavra{i}" for i in range(200))
    chunks = pipeline._recursive_split(text, 50, 0, SEPARATORS)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)
    # sem overlap, juntar os chunks reconstrói o texto original
    assert " ".join(chunks) == text


def test_recursive_split_aplica_overlap():
    text = " ".join(f"palavra{i}" for i in range(200))
    overlap = 10
    base = pipeline._recursive_split(text, 50, 0, SEPARATORS)
    chunks = pipeline._recursive_split(text, 50, overlap, SEPARATORS)
    assert len(chunks) == len(base)
    assert chunks[0] == base[0]
    for i in range(1, len(chunks)):
        assert chunks[i] == base[i - 1][-overlap:] + " " + base[i]


def test_recursive_split_palavra_maior_que_chunk_divide_por_caractere():
    text = "x" * 25
    chunks = pipeline._recursive_split(text, 10, 0, SEPARATORS)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


# ── _chunk_documents ───────────────────────────────────────────────────

def test_chunk_documents_preserva_metadata_e_indexa(monkeypatch):
    monkeypatch.setattr(pipeline, "CHUNK_SIZE", 40)
    monkeypatch.setattr(pipeline, "CHUNK_OVERLAP", 0)
    doc = Document(
        page_content=" ".join(f"termo{i}" for i in range(50)),
        metadata={"filename": "paper.pdf", "page": 3},
    )

    chunks = pipeline._chunk_documents([doc])

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert len(chunk.page_content) <= 40
        assert chunk.metadata["filename"] == "paper.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)
    # o metadata original não é alterado
    assert "chunk_index" not in doc.metadata


def test_chunk_documents_ignora_chunks_vazios():
    assert pipeline._chunk_documents([Document(page_content="   ")]) == []


# ── ingest_pdf ─────────────────────────────────────────────────────────

def test_ingest_pdf_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_pdf(tmp_path / "nao_existe.pdf")


def test_ingest_pdf_extensao_invalida(tmp_path):
    txt = tmp_path / "notas.txt"
    txt.write_text("conteúdo")
    with pytest.raises(ValueError, match=r"\.txt"):
        pipeline.ingest_pdf(txt)


def test_ingest_pdf_fluxo_completo_com_mocks(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "exemplo.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Attention is all you need. Transformers are great.")
        pdf.new_page()  # página em branco, deve ser ignorada
        pdf.save(pdf_path)

    captured = {}

    def fake_embed(chunks):
        captured["chunks"] = chunks
        return [(c, [0.0]) for c in chunks]

    monkeypatch.setattr(pipeline, "embed_documents", fake_embed)
    monkeypatch.setattr(pipeline, "add_documents", lambda items: captured.setdefault("added", items))

    result = pipeline.ingest_pdf(pdf_path)

    assert result == {"filename": "exemplo.pdf", "pages": 1, "chunks": 1}
    chunk = captured["chunks"][0]
    assert "Attention" in chunk.page_content
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["total_pages"] == 2
    assert len(captured["added"]) == 1
