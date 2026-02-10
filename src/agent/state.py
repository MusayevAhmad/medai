"""
Agent state definition for the LangGraph medical research agent.

The state flows through the graph and accumulates data as nodes execute.
``messages`` uses LangGraph's built-in message accumulator; ``citations``
uses ``operator.add`` so each tool-execution node *appends* new citations
rather than overwriting.
"""

from operator import add
from typing import Annotated, Any, Dict, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for the medical research agent.

    Attributes:
        messages: Conversation history managed by LangGraph's message
            accumulator (automatically merges new messages).
        question: The original user question (set once at invocation).
        citations: Structured citation dicts accumulated across tool calls.
            Each entry has keys: source_file, page_number, chunk_id, score,
            text_preview, extracted_entities.
    """

    messages: Annotated[list, add_messages]
    question: str
    citations: Annotated[List[Dict[str, Any]], add]
