# Research Agent

Agente de pesquisa autônomo construído com LangChain e LangGraph, capaz de
responder perguntas complexas combinando busca em documentos indexados (RAG)
e busca na web de forma dinâmica.

## Visão Geral

Este projeto implementa um agente ReAct (Reasoning + Acting) que decide
autonomamente quais ferramentas usar para responder cada pergunta — buscando
em documentos indexados, pesquisando na web ou sumarizando conteúdo longo
conforme necessário.

```
Pergunta → [Thought] → [Action] → [Observation] → ... → Resposta Final
```

## Arquitetura

```
research-agent/
├── agent/
│   ├── agent.py          # definição do agente e ferramentas
│   └── prompt.py         # system prompt do agente
├── rag/
│   ├── embedder.py       # geração de embeddings
│   └── store.py          # integração com Qdrant
├── examples/
│   └── run_agent.py      # exemplo de uso completo
├── pipeline.py           # ingestão de documentos
├── models.py             # estruturas de dados
└── config.py             # configurações centralizadas
```

## Ferramentas do Agente

| Ferramenta | O que faz | Quando o agente usa |
|---|---|---|
| `rag_search` | Busca nos documentos indexados no Qdrant | Perguntas sobre conteúdo dos papers indexados |
| `web_search` | Busca informações atuais via DuckDuckGo | Informações não disponíveis nos documentos |
| `summarize` | Sumariza textos longos | Quando o conteúdo recuperado é extenso demais |

## Padrão ReAct

```
Pergunta: "What are the most recent improvements to Transformer in 2024?"

[Thought] Esta pergunta requer informações atuais
[Action]  web_search("Transformer architecture improvements 2024")
[Observation] Resultados sobre FlashAttention e eficiência...
[Thought] O resultado é longo, melhor sumarizar
[Action]  summarize(resultado)
[Observation] Resumo conciso sobre FlashAttention...
[Thought] Tenho informação suficiente para responder
[Answer]  "Os principais avanços incluem FlashAttention..."
```

## Tecnologias

- `Python 3.10+`
- `LangChain 1.2` + `LangGraph 1.1` — orquestração do agente ReAct
- `Groq` — inferência LLM gratuita (`llama-3.3-70b-versatile`)
- `Qdrant` — vector store para busca semântica
- `sentence-transformers` — embeddings (`all-MiniLM-L6-v2`)
- `DuckDuckGo` — busca web gratuita e sem API key
- `Docker` — execução local do Qdrant

## Como Executar

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/research-agent.git
cd research-agent
```

**2. Crie e ative o ambiente virtual**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Configure as variáveis de ambiente**
```bash
# crie o arquivo .env na raiz do projeto
GROQ_API_KEY=sua_chave_aqui
```

Obtenha uma API key gratuita em `https://console.groq.com`.

**5. Suba o Qdrant com Docker**
```bash
# Windows
docker run --name qdrant-research -p 6333:6333 -v D:\caminho\do\projeto\qdrant_storage:/qdrant/storage qdrant/qdrant

# Linux/Mac
docker run --name qdrant-research -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Nas próximas vezes:
```bash
docker start qdrant-research
```

**6. Execute o exemplo**
```bash
python -m examples.run_agent
```

## Uso

```python
from pipeline import ingest_pdf
from agent import run_agent

# indexa um documento
ingest_pdf("data/paper.pdf")

# faz uma pergunta ao agente
response = run_agent("What is the main contribution of this paper?")

# resposta final
print(response.answer)

# raciocínio do agente passo a passo
for step in response.steps:
    print(f"Tool: {step['tool']}")
    print(f"Input: {step['input']}")
    print(f"Observation: {step['observation']}")

# fontes utilizadas
for source in response.sources:
    print(source)
```

## Decisões de Design

**Por que LangGraph em vez da API antiga do LangChain?**
O LangGraph é a forma recomendada de criar agentes nas versões mais recentes
do LangChain. É mais robusto, suporta streaming e oferece melhor controle
sobre o fluxo de execução do agente.

**Por que o agente decide qual ferramenta usar?**
A descrição de cada ferramenta funciona como um prompt — o LLM lê as
descrições e decide qual ferramenta é mais adequada para cada situação.
Isso é prompt engineering aplicado à orquestração de ferramentas.

**Por que `temperature=0.0`?**
Agentes de pesquisa precisam ser determinísticos e factuais. Temperatura
zero reduz alucinações e garante que o agente siga o raciocínio mais
provável em vez de introduzir criatividade desnecessária.

## Limitações Conhecidas

**Dependência de internet** — a ferramenta `web_search` requer conexão ativa.
Em ambientes offline o agente cai automaticamente para o `rag_search`.

**Qualidade da busca web** — o DuckDuckGo pode retornar resultados
inconsistentes dependendo da query. Em produção recomenda-se usar APIs
de busca mais robustas como Google Search ou Bing.

**Contexto limitado por sessão** — o agente não mantém memória entre
execuções distintas. Cada chamada a `run_agent()` é independente.