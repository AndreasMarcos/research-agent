from types import SimpleNamespace

import pytest

from rag import embedder, store


class _FakeVector(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            return _FakeVector([0.1, 0.2, 0.3])
        return [_FakeVector([float(i)]) for i, _ in enumerate(texts)]


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: model)
    return model


# ── embedder ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_embed_query_vazia_levanta_erro(fake_model, query):
    with pytest.raises(ValueError):
        embedder.embed_query(query)
    assert fake_model.calls == []


def test_embed_query_retorna_lista(fake_model):
    assert embedder.embed_query("transformer") == [0.1, 0.2, 0.3]


def test_embed_documents_vazio_nao_carrega_modelo(monkeypatch):
    def boom():
        raise AssertionError("modelo não deveria ser carregado")

    monkeypatch.setattr(embedder, "_get_model", boom)
    assert embedder.embed_documents([]) == []


# ── store.query_as_string ──────────────────────────────────────────────

class _FakeClient:
    def __init__(self, points):
        self._points = points
        self.kwargs = None

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(points=self._points)


def test_query_as_string_formata_resultados(fake_model, monkeypatch):
    points = [
        SimpleNamespace(score=0.912345, payload={"filename": "a.pdf", "page_content": "conteúdo A"}),
        SimpleNamespace(score=0.5, payload={"source": "data/b.pdf", "page_content": "conteúdo B"}),
        SimpleNamespace(score=0.1, payload=None),
    ]
    client = _FakeClient(points)
    monkeypatch.setattr(store, "_get_client", lambda: client)

    output = store.query_as_string("atenção", n_results=3)

    assert output == (
        "[1] (score=0.9123, source=a.pdf)\nconteúdo A"
        "\n\n---\n\n"
        "[2] (score=0.5, source=data/b.pdf)\nconteúdo B"
        "\n\n---\n\n"
        "[3] (score=0.1, source=desconhecido)\n"
    )
    assert client.kwargs["limit"] == 3
    assert client.kwargs["query"] == [0.1, 0.2, 0.3]
    assert client.kwargs["with_payload"] is True


def test_query_as_string_sem_resultados(fake_model, monkeypatch):
    monkeypatch.setattr(store, "_get_client", lambda: _FakeClient([]))
    assert store.query_as_string("nada") == "Nenhum resultado encontrado nos documentos indexados."
