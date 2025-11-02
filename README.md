# wiki-rag-chainlit-llamaindex
Chainlit app that builds a LlamaIndex over selected Wikipedia pages and answers questions using DeepSeek via OpenRouter.python, chainlit, llamaindex, rag, openrouter, deepseek, wikipedia, sentence-transformers, embeddings

# RAG Wikipedia Chat (Chainlit + LlamaIndex)

Chat app that indexes requested Wikipedia pages and answers questions using LlamaIndex and DeepSeek models via OpenRouter, with a Chainlit UI.

## Requirements
- Python 3.10–3.12
- pip
- OpenRouter API key

## Quick Start (Windows PowerShell)
1) Create venv and activate
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`

2) Upgrade pip and install deps
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3) Set your OpenRouter API key (choose one)
   - Current shell only: `$env:OPENROUTER_API_KEY = "sk-or-..."`
   - Persist for future shells: `setx OPENROUTER_API_KEY "sk-or-..."` (then open a new terminal)
   - Or edit `usercode/apikeys.yml` and set `openrouter_api_key: "sk-or-..."`

4) Run the app
   - `chainlit run usercode/chat_agent.py -w`
   - Open the URL shown (typically http://localhost:8000)

5) In the Chainlit settings panel
   - Select a DeepSeek model (e.g., `deepseek/deepseek-chat`)
   - Enter topics to index, e.g.: `Please index: Natural language processing, Neural networks`

## Quick Start (macOS/Linux)
1) Create venv and activate
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`

2) Upgrade pip and install deps
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3) Set your OpenRouter API key (choose one)
   - `export OPENROUTER_API_KEY="sk-or-..."`
   - Or edit `usercode/apikeys.yml`

4) Run
   - `chainlit run usercode/chat_agent.py -w`

## What’s Included
- `usercode/chat_agent.py`: Chainlit UI hooks and chat loop
- `usercode/index_wikipages.py`: Wikipedia fetching, chunking, embeddings, index build
- `usercode/utils.py`: API key loading from env or YAML
- `requirements.txt`: dependency list

## Notes
- First run downloads the embedding model `sentence-transformers/all-MiniLM-L6-v2` and may take a few minutes.
- If `torch` installation fails, install a CPU wheel from PyPI or follow official PyTorch guidance for your platform.
- If OpenRouter returns 401/403, verify the API key and outbound HTTPS access.
- Wikipedia pages that are ambiguous or missing will be skipped with a log message.

## Verify API Key (optional)
- `python usercode/utils.py` — prints a masked key if found.


