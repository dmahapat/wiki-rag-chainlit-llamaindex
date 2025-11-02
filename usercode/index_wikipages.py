"""Wikipedia indexing script (imports only for setup).

This file will later implement logic to fetch Wikipedia pages,
chunk them, embed with Hugging Face, and index via LlamaIndex.
"""

# Third-party libraries
import requests
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter

from pydantic import BaseModel

from openai import OpenAI

# Local utilities
from utils import get_apikey

import json
import traceback
from typing import List, Optional


class WikiPageList(BaseModel):
    """Validated list of Wikipedia page titles.

    Example JSON the LLM should return:
    {"pages": ["Natural language processing", "Neural networks"]}
    """

    pages: List[str] = []


def call_deepseek_api_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model:str,
) -> Optional[str]:
    """Call OpenRouter with DeepSeek and return the raw JSON content string.

    Uses model `deepseek/deepseek-r1:free` and requests a JSON-formatted response.
    Returns the assistant message content (string) or None if the call fails.
    """
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            # Encourage strict JSON output
            "response_format": {"type": "json_object"},
        }
        print(f"[llm] Calling OpenRouter DeepSeek model {model}")
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            body = (resp.text or "")
            print(f"[llm][http-{resp.status_code}] Response body (truncated): {body[:400]}")
            return None
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            print("[llm] No choices returned from API response.")
            return None
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            print("[llm] First choice message content missing or not a string.")
            return None
        return content
    except Exception as exc:
        print(f"[error] LLM call failed: {exc}")
        traceback.print_exc()
        return None


def _extract_json_text(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object from a text block.

    Handles accidental code fences or prose around JSON. Returns None if not found.
    """
    if not text:
        print("[parse] Empty text when extracting JSON.")
        return None
    # Fast path: looks like a JSON object already
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Try to find the first JSON object braces region
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    print("[parse] Could not find JSON object braces in LLM output.")
    return None


def wikipage_list(query: str, model:str) -> List[str]:
    """Return a validated list of Wikipedia page titles from an LLM response.

    - Gets OpenRouter API key via `get_apikey()`.
    - Calls DeepSeek on OpenRouter, instructing it to reply with strict JSON.
    - Parses and validates into `WikiPageList`. On failure, returns [].

    Expected queries like: "Please index: Natural language processing, Neural networks".
    """
    print(f"[wikipage_list] Query: {query}")
    api_key = get_apikey()
    if not api_key:
        print("[wikipage_list] Missing OpenRouter API key.")
        return []

    system_prompt = (
        "You are a strict JSON formatter. Respond with ONLY a JSON object. "
        "Do not include any explanations, code fences, or extra text. "
        "Schema: {\"pages\": [\"<Wikipedia page title>\", ...]}"
    )

    user_prompt = (
        "Extract the list of Wikipedia page titles to index from this request. "
        "Return only JSON in the schema above. Request: " + str(query)
    )

    raw = call_deepseek_api_openrouter(system_prompt, user_prompt, api_key, model)
    if not raw:
        print("[wikipage_list] No raw content returned from LLM.")
        return []

    json_text = _extract_json_text(raw)
    if not json_text:
        preview = (raw or "")[:400]
        print(f"[wikipage_list] Failed to extract JSON from: {preview}")
        return []

    try:
        payload = json.loads(json_text)
        # Accept common alias "titles" if present
        if isinstance(payload, dict) and "pages" not in payload and "titles" in payload:
            payload = {"pages": payload["titles"]}
        model = WikiPageList(**payload)
        # Normalize: strip and deduplicate while preserving order
        seen = set()
        result: List[str] = []
        for t in model.pages:
            if not isinstance(t, str):
                continue
            s = t.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)
        return result
    except Exception as exc:
        print(f"[parse_error] Failed to parse/validate WikiPageList: {exc}")
        traceback.print_exc()
        return []


__all__ = [
    "requests",
    "wikipedia",
    "DisambiguationError",
    "PageError",
    "HuggingFaceEmbedding",
    "VectorStoreIndex",
    "Settings",
    "Document",
    "SentenceSplitter",
    "BaseModel",
    "get_apikey",
    "OpenAI",
    # New public symbols for Task 3
    "WikiPageList",
    "wikipage_list",
    "call_deepseek_api_openrouter",
]


# -----------------------------
# Task 4: Create Documents
# -----------------------------

def safe_load_documents(titles: List[str]) -> List[Document]:
    """Fetch Wikipedia pages and build LlamaIndex Document objects.

    - Skips empty results and logs helpful debug messages.
    - Adds metadata: {"title": <resolved title>, "source": "wikipedia", "url": <page url>}.
    """
    docs: List[Document] = []
    if not titles:
        return docs

    for title in titles:
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            page = wikipedia.page(title, auto_suggest=False)
            content = getattr(page, "content", "") or ""
            if not isinstance(content, str) or not content.strip():
                print(f"[warn] Empty content for page: {title}")
                continue
            md = {
                "title": getattr(page, "title", None) or title.strip(),
                "source": "wikipedia",
                "url": getattr(page, "url", None),
            }
            docs.append(Document(text=content, metadata=md))
        except DisambiguationError as e:
            opts = ", ".join(list(getattr(e, "options", [])[:5]))
            suffix = " ..." if getattr(e, "options", []) and len(e.options) > 5 else ""
            print(f"[disambiguation] '{title}' is ambiguous. Options: {opts}{suffix}")
        except PageError:
            print(f"[missing] Page not found: {title}")
        except Exception as exc:
            print(f"[error] Failed to load '{title}': {exc}")
            traceback.print_exc()

    return docs


def create_wikidocs(wikipage_requests: WikiPageList) -> List[Document]:
    """Create Document objects from a WikiPageList instance.

    Returns a list of Document objects ready for splitting and indexing.
    """
    try:
        titles = list(wikipage_requests.pages) if getattr(wikipage_requests, "pages", None) else []
    except Exception as exc:
        print(f"[error] Could not read pages from WikiPageList: {exc}")
        traceback.print_exc()
        titles = []
    return safe_load_documents(titles)


# Export Task 4 helpers
__all__.extend([
    "safe_load_documents",
    "create_wikidocs",
])


# -----------------------------
# Task 5: Creating the Index
# -----------------------------

# Global index variable as requested
index: Optional[VectorStoreIndex] = None


def create_index(query: str, model: str) -> Optional[VectorStoreIndex]:
    """Create a vector index from a user query of Wikipedia pages.

    Steps:
    1) Extract page titles via `wikipage_list(query)`.
    2) Fetch and wrap content via `create_wikidocs()`.
    3) Split into nodes with `SentenceSplitter().get_nodes_from_documents()`.
    4) Load embeddings via HuggingFaceEmbedding and set `Settings.embed_model`.
    5) Build `VectorStoreIndex` and store in global `index`.
    """
    global index

    print(f"[index] Creating index from query: {query}")
    titles = wikipage_list(query, model)
    if not titles:
        print("[info] No Wikipedia titles extracted from query.")
        index = None
        return None

    print(f"[index] Titles extracted: {titles}")
    docs = create_wikidocs(WikiPageList(pages=titles))
    if not docs:
        print("[info] No documents were created from the provided titles.")
        index = None
        return None

    try:
        splitter = SentenceSplitter()
        nodes = splitter.get_nodes_from_documents(docs)
        if not nodes:
            print("[info] No nodes produced by splitter; cannot build index.")
            index = None
            return None

        print("[index] Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model

        print(f"[index] Building VectorStoreIndex with {len(nodes)} nodes…")
        index = VectorStoreIndex(nodes)
        print(f"[ok] Index created with {len(nodes)} nodes from {len(docs)} documents.")
        return index
    except Exception as exc:
        print(f"[error] Failed to create index: {exc}")
        traceback.print_exc()
        index = None
        return None


__all__.extend([
    "index",
    "create_index",
])


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]).strip()
    if not user_query:
        print("Usage: python usercode/index_wikipages.py \"please index: London, Tokyo\"")
        sys.exit(2)
    created = create_index(user_query)
    sys.exit(0 if created is not None else 1)
