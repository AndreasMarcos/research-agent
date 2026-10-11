import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from agent.prompt import SYSTEM_PROMPT
from models import AgentResponse
from config import GROQ_API_KEY, GROQ_MODEL
from tools import rag_search, web_search, summarize

logger = logging.getLogger(__name__)


# ── agente ─────────────────────────────────────────────────────────────

_agent = None


def _get_agent():
    """Inicializa o agente uma única vez (singleton)."""
    global _agent
    if _agent is None:
        logger.info("Inicializando agente...")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.0,
        )
        tools = [rag_search, web_search, summarize]
        _agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )
    return _agent


def _parse_messages(messages: list) -> AgentResponse:
    """
    Converte as mensagens retornadas pelo agente em um AgentResponse.

    Args:
        messages: lista de mensagens do LangGraph (resultado de agent.invoke)

    Returns:
        AgentResponse com resposta final, steps e sources.
    """
    # extrai resposta final — última mensagem do agente
    answer = messages[-1].content if messages else ""

    # extrai steps e sources das mensagens intermediárias
    steps = []
    sources = []

    for msg in messages:
        # mensagens de tool use
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                steps.append({
                    "tool": tool_call["name"],
                    "input": tool_call["args"].get("query", tool_call["args"]),
                    "observation": "",
                })

        # resultados das ferramentas
        if hasattr(msg, "name") and msg.name:
            if steps and not steps[-1]["observation"]:
                steps[-1]["observation"] = str(msg.content)[:500]

            if msg.name == "rag_search":
                for line in str(msg.content).splitlines():
                    if "source=" in line:
                        source = line.split("source=")[-1].strip(")")
                        if source and source not in sources:
                            sources.append(source)

    return AgentResponse(
        answer=answer,
        steps=steps,
        sources=sources,
    )


def run_agent(question: str) -> AgentResponse:
    """
    Executa o agente para responder uma pergunta.

    Args:
        question: pergunta em linguagem natural

    Returns:
        AgentResponse com resposta, steps e sources.
    """
    logger.info("Pergunta: '%s'", question)

    agent = _get_agent()

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
    except Exception as e:
        logger.error("Erro ao executar o agente: %s", e)
        return AgentResponse(
            answer=f"Erro ao executar o agente: {e}",
            steps=[],
            sources=[],
        )

    return _parse_messages(result.get("messages", []))
