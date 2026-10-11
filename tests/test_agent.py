from types import SimpleNamespace

from agent.agent import _parse_messages
from models import AgentResponse


def _human(content):
    return SimpleNamespace(content=content)


def _ai(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _tool(name, content):
    return SimpleNamespace(content=content, name=name)


def test_parse_messages_vazio():
    assert _parse_messages([]) == AgentResponse(answer="", steps=[], sources=[])


def test_parse_messages_sem_ferramentas():
    response = _parse_messages([_human("Oi?"), _ai("Olá!")])
    assert response.answer == "Olá!"
    assert response.steps == []
    assert response.sources == []


def test_parse_messages_extrai_steps_e_sources():
    rag_output = (
        "[1] (score=0.9, source=attention.pdf)\ntexto 1"
        "\n\n---\n\n"
        "[2] (score=0.8, source=bert.pdf)\ntexto 2"
        "\n\n---\n\n"
        "[3] (score=0.7, source=attention.pdf)\ntexto 3"
    )
    messages = [
        _human("Pergunta"),
        _ai(tool_calls=[{"name": "rag_search", "args": {"query": "attention"}}]),
        _tool("rag_search", rag_output),
        _ai(tool_calls=[{"name": "summarize", "args": {"text": "longo"}}]),
        _tool("summarize", "resumo"),
        _ai("Resposta final"),
    ]

    response = _parse_messages(messages)

    assert response.answer == "Resposta final"
    assert response.steps == [
        {"tool": "rag_search", "input": "attention", "observation": rag_output},
        # sem "query" nos args, o input é o dicionário de args inteiro
        {"tool": "summarize", "input": {"text": "longo"}, "observation": "resumo"},
    ]
    # sources deduplicados e na ordem de aparição
    assert response.sources == ["attention.pdf", "bert.pdf"]


def test_parse_messages_trunca_observation_em_500():
    messages = [
        _ai(tool_calls=[{"name": "web_search", "args": {"query": "q"}}]),
        _tool("web_search", "x" * 1200),
        _ai("fim"),
    ]
    response = _parse_messages(messages)
    assert response.steps[0]["observation"] == "x" * 500


def test_parse_messages_sources_apenas_de_rag_search():
    messages = [
        _ai(tool_calls=[{"name": "web_search", "args": {"query": "q"}}]),
        _tool("web_search", "(score=1, source=site.com)"),
        _ai("fim"),
    ]
    assert _parse_messages(messages).sources == []


def test_parse_messages_com_mensagens_reais_do_langchain():
    import pytest
    messages_mod = pytest.importorskip("langchain_core.messages")

    messages = [
        messages_mod.HumanMessage(content="Pergunta"),
        messages_mod.AIMessage(
            content="",
            tool_calls=[{"name": "rag_search", "args": {"query": "bleu"}, "id": "call_1"}],
        ),
        messages_mod.ToolMessage(
            content="[1] (score=0.9, source=attention.pdf)\nBLEU 28.4",
            name="rag_search",
            tool_call_id="call_1",
        ),
        messages_mod.AIMessage(content="O BLEU foi 28.4."),
    ]

    response = _parse_messages(messages)

    assert response.answer == "O BLEU foi 28.4."
    assert response.steps[0]["tool"] == "rag_search"
    assert response.steps[0]["input"] == "bleu"
    assert response.sources == ["attention.pdf"]
