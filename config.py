from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ── Groq API ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Embedding ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Qdrant ─────────────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "research_agent"

# ── Agent ──────────────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS = 10     # limite de ciclos ReAct para evitar loop infinito
AGENT_VERBOSE = True          # exibe o raciocínio do agente passo a passo

# ── RAG Tool ───────────────────────────────────────────────────────────
RAG_TOP_K = 5                 # número de chunks retornados por busca

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64