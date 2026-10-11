import logging

from langchain_core.tools import tool
from rag.store import query_as_string

logger = logging.getLogger(__name__)


@tool
def rag_search(query: str) -> str:
    """Search for information in indexed scientific documents.
    Use this tool when the question is about content that may be in
    the indexed documents such as research papers, methodologies,
    datasets or specific technical details."""
    logger.info("Buscando: '%s'", query)
    return query_as_string(query)
