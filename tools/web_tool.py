import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search for current information on the web using DuckDuckGo.
    Use this tool when the question requires up-to-date information
    or information not available in the indexed documents."""
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    logger.info("Buscando na web: '%s'", query)
    try:
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)
    except Exception as e:
        logger.error("Erro na busca web: %s", e)
        return f"Erro na busca web: {e}"
