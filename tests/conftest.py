"""
Configuração compartilhada dos testes.

- Coloca a raiz do repositório no sys.path.
- Substitui por stubs os pacotes pesados (sentence_transformers, langchain_groq,
  langgraph) quando não estão instalados, para que os testes rodem offline,
  sem API key, sem servidor Qdrant e sem download de modelos.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _stub_module(name: str, **attrs) -> None:
    """Registra um módulo falso em sys.modules (e seus pacotes pai)."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        if i > 1:
            setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], sys.modules[mod_name])
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)


class _Unavailable:
    """Placeholder que falha se algum teste tentar usá-lo de verdade."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Dependência pesada indisponível nos testes (stub).")


if not _is_installed("sentence_transformers"):
    _stub_module("sentence_transformers", SentenceTransformer=_Unavailable)

if not _is_installed("langchain_groq"):
    _stub_module("langchain_groq", ChatGroq=_Unavailable)

if not _is_installed("langgraph"):
    _stub_module("langgraph.prebuilt", create_react_agent=_Unavailable)
