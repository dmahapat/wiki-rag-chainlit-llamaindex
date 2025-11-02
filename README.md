# wiki-rag-chainlit-llamaindex
Chainlit app that builds a LlamaIndex over selected Wikipedia pages and answers questions using DeepSeek via OpenRouter.python, chainlit, llamaindex, rag, openrouter, deepseek, wikipedia, sentence-transformers, embeddings

# RAG Wikipedia Chat (Chainlit + LlamaIndex)

Chat app that indexes requested Wikipedia pages and answers questions using LlamaIndex and DeepSeek models via OpenRouter, with a Chainlit UI.

---

## Requirements
- Python 3.10–3.12
- pip
- OpenRouter API key

Why these?
- Python version range ensures compatibility with Chainlit, LlamaIndex, and dependencies.
- pip installs the pinned dependencies in `requirements.txt`.
- The OpenRouter API key lets the app call DeepSeek models (via OpenRouter) to help choose Wikipedia pages to index and to respond to questions.

---

## First-Time Setup — Detailed (Windows PowerShell)

1) Create and activate a virtual environment
   - Command:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
   - What this does: Creates an isolated Python environment so installed packages don’t affect your global Python. Activation ensures `python`/`pip` map to the venv.

2) Upgrade pip and install dependencies
   - Command:
     - `python -m pip install --upgrade pip`
     - `pip install -r requirements.txt`
   - What this does: Upgrades your package manager, then installs all required libraries (Chainlit, LlamaIndex, sentence-transformers, torch, etc.). First run may download model weights later.

3) Provide your OpenRouter API key (choose one)
   - Environment variable (recommended for local dev):
     - Current shell only: `$env:OPENROUTER_API_KEY = "sk-or-..."`
     - Persist for new shells: `setx OPENROUTER_API_KEY "sk-or-..."` then open a new terminal
   - Or YAML file (checked by `usercode/utils.py`):
     - Edit `usercode/apikeys.yml` and set `openrouter_api_key: "sk-or-..."`
   - What this does: Makes the key available to calls to OpenRouter. The app reads from the env var first, then falls back to the YAML file.

4) Run the app
   - Command:
     - `chainlit run usercode/chat_agent.py -w`
   - What this does: Launches Chainlit’s dev server with autoreload (`-w`). The terminal prints a local URL (usually http://localhost:8000). Open it in a browser.

5) Configure in the Chainlit UI
   - In the settings panel:
     - Choose a model (e.g., `deepseek/deepseek-chat`).
     - Enter topics to index (e.g., `Please index: Natural language processing, Neural networks`).
   - What this does: The app calls OpenRouter to produce a clean JSON list of Wikipedia page titles, fetches those pages, splits them into nodes, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, builds a `VectorStoreIndex`, and wires a query engine for chat.

---

## First-Time Setup — Detailed (macOS/Linux)

1) Create and activate a virtual environment
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`

2) Upgrade pip and install dependencies
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3) Provide your OpenRouter API key (choose one)
   - `export OPENROUTER_API_KEY="sk-or-..."`
   - Or edit `usercode/apikeys.yml`

4) Run
   - `chainlit run usercode/chat_agent.py -w`

---

## How It Works (Architecture Overview)
- UI: Chainlit (`usercode/chat_agent.py`) handles the settings dialog and chat events.
- Page selection: The app calls a DeepSeek model via OpenRouter to turn your free‑form request into a strict JSON list of Wikipedia page titles.
- Data ingestion: `wikipedia` library downloads each page’s content.
- Chunking: `SentenceSplitter` splits content into nodes suitable for retrieval.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via `llama-index-embeddings-huggingface`) converts nodes to vectors.
- Index: `VectorStoreIndex` stores vectors for similarity search.
- Query engine: Wraps the index with the selected LLM to produce grounded answers referencing the indexed content.

---

## Verify Your API Key
- `python usercode/utils.py`
- What this does: Prints a masked key if the env var or YAML is configured correctly; exits non‑zero otherwise.

---

## What’s Included
- `usercode/chat_agent.py`: Chainlit UI hooks, settings handling, chat loop.
- `usercode/index_wikipages.py`: LLM call to pick pages, Wikipedia fetch, chunking, embeddings, and index build.
- `usercode/utils.py`: Load API key from env or YAML.
- `requirements.txt`: Dependency list.
- `chainlit.md`: Optional welcome screen content for Chainlit.

---

## Optional: Push This Repo to GitHub (Explained)

1) Initialize and set branch
   - `git init`
   - `git branch -M main`
   - Why: Creates a new Git repository locally and standardizes the default branch name.

2) Keep secrets out of Git
   - `.gitignore` already ignores typical secrets and local artifacts.
   - If you want to publish an example, create `usercode/apikeys.example.yml` with `openrouter_api_key: "sk-or-REPLACE_ME"`.

3) Stage and commit
   - `git add -A`
   - `git commit -m "Initial commit: Chainlit RAG app"`
   - Why: Records a snapshot of your project in Git history.

4) Point to your remote and push
   - `git remote add origin https://github.com/<you>/<repo>.git`
   - `git push -u origin main`
   - Why: Links your local repo to GitHub and uploads your commits. Use `git pull --rebase origin main` first if the remote already has commits.

---

## Troubleshooting
- Torch install errors
  - Use a CPU wheel from PyPI, or follow the official PyTorch instructions for your OS and Python version.
- OpenRouter 401/403
  - Verify `OPENROUTER_API_KEY`. For new shells on Windows, use `setx` and restart the terminal.
- Slow first run
  - Model weights (e.g., sentence‑transformers) download on first use. Subsequent runs are faster.
- Wikipedia ambiguity or missing pages
  - The app logs disambiguation or missing‑page messages and skips those titles. Try more specific titles.
