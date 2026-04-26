from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rag.store import query_as_string
from agent.prompt import SYSTEM_PROMPT
from models import AgentResponse
from config import GROQ_API_KEY, GROQ_MODEL


# ── definição das ferramentas ──────────────────────────────────────────

@tool
def rag_search(query: str) -> str:
    """Search for information in indexed scientific documents.
    Use this tool when the question is about content that may be in
    the indexed documents such as research papers, methodologies,
    datasets or specific technical details."""
    print(f"\n[rag_tool] Buscando: '{query}'")
    return query_as_string(query)


@tool
def web_search(query: str) -> str:
    """Search for current information on the web using DuckDuckGo.
    Use this tool when the question requires up-to-date information
    or information not available in the indexed documents."""
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    print(f"\n[web_tool] Buscando na web: '{query}'")
    try:
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Erro na busca web: {e}"


@tool
def summarize(text: str) -> str:
    """Summarize a long text into a concise paragraph.
    Use this tool when the retrieved content is too long or verbose
    and needs to be condensed before including in the final answer."""
    from groq import Groq
    print(f"\n[summarize_tool] Sumarizando {len(text)} caracteres...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Summarize the following text concisely and accurately."},
                {"role": "user", "content": f"Summarize:\n\n{text}"}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao sumarizar: {e}"


# ── agente ─────────────────────────────────────────────────────────────

_agent = None


def _get_agent():
    """Inicializa o agente uma única vez (singleton)."""
    global _agent
    if _agent is None:
        print("[agent] Inicializando agente...")
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


def run_agent(question: str) -> AgentResponse:
    """
    Executa o agente para responder uma pergunta.

    Args:
        question: pergunta em linguagem natural

    Returns:
        AgentResponse com resposta, steps e sources.
    """
    print(f"\n[agent] Pergunta: '{question}'")

    agent = _get_agent()

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
    except Exception as e:
        return AgentResponse(
            answer=f"Erro ao executar o agente: {e}",
            steps=[],
            sources=[],
        )

    # extrai resposta final — última mensagem do agente
    messages = result.get("messages", [])
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