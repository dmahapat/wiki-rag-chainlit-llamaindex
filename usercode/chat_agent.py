"""Chat agent setup imports for Chainlit + LlamaIndex.

This file currently contains imports only. Upcoming tasks will
wire these pieces together to initialize the conversational agent.
"""

# UI framework and input widgets
import chainlit as cl
from chainlit.input_widget import Select, TextInput

# LLM integration for LlamaIndex via OpenAI-compatible interface
from llama_index.llms.openai import OpenAI

# Tools and agent components from LlamaIndex
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core import PromptTemplate

from llama_index.llms.openrouter import OpenRouter

from llama_index.core.tools.query_engine import QueryEngineTool




# Local functions
from index_wikipages import create_index
from utils import get_apikey


__all__ = [
    "cl",
    "Select",
    "TextInput",
    "OpenAI",
    "QueryEngineTool",
    "ToolMetadata",
    "PromptTemplate",
    "ReActAgent",
    "create_index",
    "get_apikey",
]


# -----------------------------
# Task 7: Initialize the Settings Menu
# -----------------------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("settings_ready", False)
    """Show settings to pick model and Wikipedia topic at session start."""
    await cl.ChatSettings(
        [
            Select(
                id="deepseek_model",
                label="DeepSeek - Model",
                values=[
                    "deepseek/deepseek-chat",
                    "deepseek/deepseek-chat:free",
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-r1:free",
                ],
                initial_value="deepseek/deepseek-chat",   
            ),
            TextInput(
                id="wikipage_request",
                label="Request Wikipage",
                placeholder="Please index: Natural language processing, Neural networks",
            ),
        ]
    ).send()

    await cl.Message(
        content=(
            "Welcome! Please choose a DeepSeek model and enter the Wikipedia topic(s) "
            "to index before chatting."
        )
    ).send()


# -----------------------------
# Task 8: Wikipedia Search Engine
# -----------------------------

def wikisearch_engine(index, llm):
    """Return a query engine over the provided index using the given LLM.

    Configures concise responses, verbose logging, and top-3 similarity retrieval.
    """
    return index.as_query_engine(
        response_mode="compact",
        verbose=False,
        similarity_top_k=3,
        llm=llm,
    )


__all__.extend([
    "wikisearch_engine",
])


# -----------------------------
# Task 9: Create the ReAct Agent
# -----------------------------

def create_react_agent(MODEL: str, index):
    if index is None:
        print("[info] No index available to build the agent.")
        return None

    selected_model = MODEL.strip()
    print(f"[LLM] Initializing OpenRouter model → {selected_model}")

    llm = OpenRouter(
        model=selected_model,
        api_key=get_apikey()
    )

    qe = wikisearch_engine(index, llm)

    tool = QueryEngineTool.from_defaults(
    query_engine=qe,
    name="wikipedia_search",
    description=(
        "Semantic search over the indexed Wikipedia pages. "
        "Use this to answer questions grounded in those pages."
    ),
    )

    class SimpleAgent:
        def __init__(self, qe):
            self._qe = qe

        def chat(self, text: str):
            # text must be a plain string, not a Chainlit Message
            return self._qe.query(text)

    return SimpleAgent(qe) 



    #agent = ReActAgent.from_tools([tool], llm=llm, verbose=False)
    #worker = ReActAgentWorker.from_tools([tool], llm=llm, verbose=False)
    #agent = AgentRunner(worker)
    #return agent



__all__.extend([
    "create_react_agent",
])


# -----------------------------
# Task 10: Finalize the Settings Menu
# -----------------------------

@cl.on_settings_update
async def setup_agent(settings: dict):
    """Build a fresh index and agent whenever settings are updated.

    Expects settings keys for model and request topic. Supports both the
    IDs defined in this file ("deepseek_model", "wikipage_request") and
    alternative names ("MODEL", "WikiPageRequest").
    """

    if not cl.user_session.get("settings_ready"):
        cl.user_session.set("settings_ready", True)
        return
    
    # Extract model and wiki request with flexible keys
    model = (
        (settings.get("MODEL") or settings.get("model") or settings.get("deepseek_model") or "").strip()
        if isinstance(settings, dict)
        else ""
    )
    request = (
        (settings.get("WikiPageRequest") or settings.get("wikipage_request") or "").strip()
        if isinstance(settings, dict)
        else ""
    )

    if not model:
        await cl.Message(content="Please select a model in settings." ).send()
        return
    if not request:
        await cl.Message(content="Please enter a Wikipedia request in settings." ).send()
        return

    await cl.Message(content=f"Indexing requested pages from: {request}\nUsing model: {model}" ).send()

    # Build vector index from request
    idx = create_index(request, model)

    # Initialize ReAct agent
    agent = create_react_agent(model, idx) if idx is not None else None

    if agent is not None and idx is not None:
        # Stash in session for later message handling
        cl.user_session.set("agent", agent)
        await cl.Message(content="Wikipedia pages indexed and agent is ready!" ).send()
    else:
        await cl.Message(content="Failed to initialize index or agent. Please adjust settings and retry." ).send()


# -----------------------------
# Task 11: Script the Chat Interactions
# -----------------------------

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    if agent is None:
        await cl.Message("Please configure settings before chatting.").send()
        return

    user_text = (message.content or "").strip()
    if not user_text:
        await cl.Message("Please type something.").send()
        return

    result = agent.chat(user_text)          # <— plain string input
    text = getattr(result, "response", None) or str(result)

    await cl.Message(text).send()

__all__.extend([
    "main",
])
