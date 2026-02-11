"""
LangGraph StateGraph definition for the medical research agent.

Builds and compiles a ReAct-style graph:

    START ──> agent ──┬──> tools ──> agent  (loop)
                      └──> END

The agent node calls the LLM with tool bindings.  If the LLM emits
tool calls, execution routes to the *tools* node and back.  When the
LLM returns a plain text answer (no tool calls), execution ends.

Usage:
    from src.agent.graph import build_agent_graph, run_agent

    graph = build_agent_graph(llm=chat_model, retriever=retriever)
    result = run_agent(graph, "Compare aspirin and ibuprofen")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agent.nodes import create_agent_node, create_tool_executor
from src.agent.state import AgentState
from src.agent.tools import create_search_tool

if TYPE_CHECKING:
    from src.retrieve import HybridRetriever

logger = logging.getLogger(__name__)

# Maximum number of graph steps (prevents infinite loops)
DEFAULT_RECURSION_LIMIT = 15


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------

def _should_continue(state: Dict) -> str:
    """Route after the agent node: tools or END.

    If the last message has ``tool_calls``, we need to execute them.
    Otherwise the agent has produced its final answer.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# -------------------------------------------------------------------
# Complexity detection (for auto-routing in /query)
# -------------------------------------------------------------------

_COMPLEX_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)?\s+between\b", re.IGNORECASE),
    re.compile(r"\bsimilarit(?:y|ies)\s+between\b", re.IGNORECASE),
    re.compile(r"\brelationship\s+between\b", re.IGNORECASE),
    re.compile(r"\binteract(?:s|ion)?\s+with\b", re.IGNORECASE),
    re.compile(r"\bboth\b.*\band\b", re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+.+\s+differ\b", re.IGNORECASE),
    re.compile(r"\bpros\s+and\s+cons\b", re.IGNORECASE),
]


def is_complex_query(question: str, entities: Optional[list] = None) -> bool:
    """Detect if a question requires multi-step agent reasoning.

    Checks for comparison patterns in the text and for multiple entities
    of the same type (e.g. two ``Chemical`` entities suggesting a drug
    comparison).

    Args:
        question: The user's question text.
        entities: Optional list of ``Entity`` objects from NER extraction.

    Returns:
        True if the question likely needs the agent pipeline.
    """
    # Pattern-based detection
    if any(p.search(question) for p in _COMPLEX_PATTERNS):
        return True

    # Entity-based detection: multiple entities of the same type
    if entities:
        from collections import Counter

        label_counts = Counter(
            getattr(e, "label", e.get("label", "")) if isinstance(e, dict) else e.label
            for e in entities
        )
        if any(count >= 2 for count in label_counts.values()):
            return True

    return False


# -------------------------------------------------------------------
# Graph builder
# -------------------------------------------------------------------

def build_agent_graph(llm: Any, retriever: HybridRetriever) -> Any:
    """Build and compile the medical research agent graph.

    Creates a ReAct-pattern StateGraph with:
    - An *agent* node (LLM with tool bindings)
    - A *tools* node (custom executor that captures citations)
    - Conditional edge: agent → tools (if tool calls) or agent → END

    Args:
        llm: LangChain-compatible chat model (e.g. ``ChatOllama``).
        retriever: Initialised ``HybridRetriever`` for guideline search.

    Returns:
        Compiled LangGraph runnable.
    """
    # Create tools
    search_tool = create_search_tool(retriever)
    tools = [search_tool]

    # Create nodes
    agent_node = create_agent_node(llm, tools)
    tool_executor = create_tool_executor(retriever, tools)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_executor)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info(
        "Agent graph compiled successfully (tools: %s)",
        [t.name for t in tools],
    )
    return compiled


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def run_agent(
    graph: Any,
    question: str,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> Dict:
    """Run the agent graph on a question and return structured results.

    Invokes the compiled graph with an initial ``HumanMessage`` and
    extracts the final answer + accumulated citations from the terminal
    state.

    Args:
        graph: Compiled LangGraph agent (from ``build_agent_graph``).
        question: The user's medical question.
        recursion_limit: Max number of graph steps to prevent runaway loops.

    Returns:
        Dict with keys:
            - ``answer`` (str): The agent's final synthesised answer.
            - ``citations`` (list[dict]): Deduplicated structured citations.
            - ``steps`` (int): Total number of messages in the trace.
            - ``messages`` (list): Full message history for observability.
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "citations": [],
    }

    config = {"recursion_limit": recursion_limit}

    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.error("Agent execution failed: %s", exc, exc_info=True)
        return {
            "answer": (
                "I encountered an error while processing your question. "
                "Please try rephrasing or using a simpler query."
            ),
            "citations": [],
            "steps": 0,
            "messages": [],
        }

    # Extract the final text answer (last AI message without tool calls)
    answer = ""
    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Skip tool messages and AI messages that only contain tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            if hasattr(msg, "type") and msg.type == "tool":
                continue
            answer = msg.content
            break

    # Deduplicate citations by chunk_id
    seen_chunks: set = set()
    unique_citations: List[Dict] = []
    for c in final_state.get("citations", []):
        cid = c.get("chunk_id", "")
        if cid and cid not in seen_chunks:
            seen_chunks.add(cid)
            unique_citations.append(c)
        elif not cid:
            unique_citations.append(c)

    return {
        "answer": answer,
        "citations": unique_citations,
        "steps": len(final_state["messages"]),
        "messages": final_state["messages"],
    }
