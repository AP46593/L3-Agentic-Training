# Tech Stack

## Language
- Python 3.14.4

## Key Libraries & Frameworks
| Category | Libraries |
|----------|-----------|
| Agent Framework | LangChain, LangChain-Core, LangGraph |
| LLM Provider | LangChain-Ollama (local models) |
| Vector Stores | FAISS (faiss-cpu), ChromaDB |
| Graph DB | Neo4j, LangChain-Neo4j |
| MCP | mcp (Model Context Protocol) |
| Observability | LangSmith |
| Document Loaders | pypdf, python-docx |
| HTTP | httpx, requests |
| Validation | Pydantic v2 |
| Config | python-dotenv |
| CLI/Output | Rich |
| Notebooks | Jupyter, ipykernel |

## Package Manager
- **uv** — fast Python package & project manager (replaces pip + venv)
- Install uv: https://docs.astral.sh/uv/getting-started/installation/

## Environment Setup
```bash
uv venv .venv
.venv\Scripts\activate       # Windows
uv pip install -r L3-Agentic-Training/requirements.txt
```

## Common Commands
| Action | Command |
|--------|---------|
| Create venv | `uv venv .venv` |
| Activate venv (Windows) | `.venv\Scripts\activate` |
| Activate venv (Linux/Mac) | `source .venv/bin/activate` |
| Install deps | `uv pip install -r L3-Agentic-Training/requirements.txt` |
| Add a package | `uv pip install <package>` |
| Run notebook | `jupyter notebook` |
| Run script | `python <script>.py` |

## Environment Variables
- Managed via `.env` files (gitignored)
- Loaded with `python-dotenv`
- Expect keys for LangSmith, Neo4j, and any LLM API providers

## Ollama Models
| Model | Size | Use Case |
|-------|------|----------|
| llama3 | 8B | General-purpose chat & reasoning |
| deepseek-r1 | 7B | Code generation & reasoning |
| phi4 | 14B | Compact reasoning, summarization |
| gpt-oss:120b-cloud | 120B | High-capability cloud model for complex tasks |

Pull models locally with:
```bash
ollama pull llama3
ollama pull deepseek-r1
ollama pull phi4
```

## Notes
- CrewAI is commented out in requirements — requires Python < 3.14
- Ollama is the primary LLM backend (local inference)
- Use smaller models (llama3, phi4) for fast local iteration; fall back to larger/cloud models for complex reasoning
