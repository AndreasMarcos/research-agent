import logging

from langchain_core.tools import tool
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


@tool
def summarize(text: str) -> str:
    """Summarize a long text into a concise paragraph.
    Use this tool when the retrieved content is too long or verbose
    and needs to be condensed before including in the final answer."""
    from groq import Groq
    logger.info("Sumarizando %d caracteres...", len(text))
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
        logger.error("Erro ao sumarizar: %s", e)
        return f"Erro ao sumarizar: {e}"
