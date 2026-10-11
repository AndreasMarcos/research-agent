# Exemplo de uso do Research Agent.
# Ingere documentos e executa perguntas complexas via agente ReAct.
#
# Execução:
#   python -m examples.run_agent

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import ingest_pdf, status
from agent import run_agent
from rag.store import delete_collection


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # ── ingestão ──────────────────────────────────────────────────────
    print("=== Resetando índice ===")
    delete_collection()

    print("\n=== Ingestão ===")
    result = ingest_pdf("data/attention.pdf")
    print(f"  {result['filename']} — {result['pages']} páginas, {result['chunks']} chunks")
    status()

    # ── perguntas ao agente ───────────────────────────────────────────
    questions = [
        "What is the main contribution of the Attention Is All You Need paper?",
        "What were the BLEU scores achieved on WMT 2014 tasks?",
        "How does multi-head attention differ from single-head attention?",
        "What are the most recent improvements to the Transformer architecture in 2024?",
    ]

    print("\n=== Agente ===")
    for question in questions:
        print(f"\n{'═'*60}")
        print(f"Pergunta: {question}")
        print('═'*60)

        response = run_agent(question)

        print(f"\n--- Steps ---")
        for i, step in enumerate(response.steps, start=1):
            print(f"  [{i}] Tool: {step['tool']}")
            print(f"       Input: {step['input']}")
            print(f"       Observation: {step['observation'][:200]}...")

        print(f"\n--- Resposta Final ---")
        print(response.answer)

        if response.sources:
            print(f"\n--- Fontes ---")
            for source in response.sources:
                print(f"  - {source}")


if __name__ == "__main__":
    main()