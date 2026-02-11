"""
Node functions for the LangGraph medical research agent.

Nodes are the building blocks of the StateGraph:

- **agent_node** — calls the LLM with tool bindings; the LLM either
  returns tool calls (routed to *tool_executor*) or a final text answer
  (routed to END).
- **tool_executor** — executes the requested tools, stores results as
  ``ToolMessage`` objects *and* captures structured citations in the
  agent state for the API response layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from langchain_core.messages import SystemMessage, ToolMessage

if TYPE_CHECKING:
    from src.retrieve import HybridRetriever

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# System prompt — instructs the agent on decomposition & citation rules
# -----------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are BioScholar, a medical research assistant with access to a database of \
clinical practice guidelines and medical literature.

## How to Answer Questions

1. **Simple questions** (single topic): Search the guidelines once, then answer.
2. **Comparison questions** ("Compare X and Y", "X vs Y"): Search for each \
item separately, then synthesise a comparison.
3. **Multi-part questions** ("What are the symptoms AND treatments for X?"): \
Search for each aspect separately, then combine into one coherent answer.

## Rules
- Base your answer ENTIRELY on the search results. Do NOT use outside knowledge.
- Cite sources using [Source N] notation from the search results.
- If the search results don't contain enough information, say so honestly.
- Be concise but thorough. Use medical terminology accurately.
- When combining results from multiple searches, present a coherent synthesised \
answer with clear section headings.
- After gathering ALL needed information, provide your final answer.
- Do NOT repeat the raw search results — summarise and synthesise them."""


def create_agent_node(llm: Any, tools: list) -> Callable:
    """Create the agent reasoning node.

    Binds *tools* to *llm* so the model can emit ``tool_calls`` in its
    response.  The system prompt is prepended to every invocation to
    ensure consistent behaviour.

    Args:
        llm: LangChain-compatible chat model (e.g. ``ChatOllama``).
        tools: List of LangChain tool objects to bind.

    Returns:
        A callable ``(state) -> state-update`` suitable for
        ``StateGraph.add_node()``.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: Dict) -> Dict:
        messages = list(state["messages"])

        # Ensure the system prompt is always the first message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def create_tool_executor(retriever: HybridRetriever, tools: list) -> Callable:
    """Create a custom tool executor that captures structured citations.

    Unlike LangGraph's prebuilt ``ToolNode``, this executor **also**
    appends structured citation dicts to ``state["citations"]`` so the
    API response layer can build ``Citation`` objects without re-running
    the search.

    For ``search_guidelines`` calls the retriever is called directly
    (avoiding a double search).  Unknown tool names are executed via the
    standard LangChain tool interface.

    Args:
        retriever: Initialised ``HybridRetriever``.
        tools: List of LangChain tool objects (for fallback execution).

    Returns:
        A callable ``(state) -> state-update`` suitable for
        ``StateGraph.add_node()``.
    """
    tools_by_name = {t.name: t for t in tools}

    def execute_tools(state: Dict) -> Dict:
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": [], "citations": []}

        tool_messages: List[ToolMessage] = []
        new_citations: List[Dict] = []

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            logger.info("Executing tool: %s(%s)", tool_name, tool_args)

            if tool_name == "search_guidelines":
                # Extract query — handle malformed args from small models
                # Some models return {'query': 'text'} (correct) while others
                # return {'query': {'content': 'text', 'type': 'string'}}
                raw_query = tool_args.get("query", "")
                if isinstance(raw_query, dict):
                    # Try to extract the actual query from nested dict
                    raw_query = (
                        raw_query.get("content")
                        or raw_query.get("value")
                        or raw_query.get("text")
                        or ""
                    )
                    logger.info(
                        "Extracted query from nested tool args: %r",
                        raw_query,
                    )
                if not isinstance(raw_query, str) or not raw_query.strip():
                    # Final fallback: use the original user question
                    raw_query = state.get("question", "")
                    logger.warning(
                        "Could not parse tool args (got %r), falling back "
                        "to original question: %r",
                        tool_args,
                        raw_query,
                    )

                # Run retriever directly and capture structured results
                result = retriever.search(
                    query=raw_query,
                    top_k=5,
                    entity_filter=True,
                )

                # Format text for the LLM
                from src.agent.tools import _format_search_results

                formatted = _format_search_results(result["results"])
                tool_messages.append(
                    ToolMessage(content=formatted, tool_call_id=tool_call_id)
                )

                # Store structured citations
                for r in result["results"]:
                    new_citations.append(
                        {
                            "source_file": r.get("source_file", ""),
                            "page_number": r.get("page_number", 0),
                            "chunk_id": r.get("chunk_id", ""),
                            "score": round(r.get("score", 0), 4),
                            "text_preview": (r.get("text", "") or "")[:200],
                            "extracted_entities": r.get(
                                "extracted_entities", []
                            ),
                        }
                    )
            else:
                # Fallback: execute via standard LangChain tool interface
                tool_fn = tools_by_name.get(tool_name)
                if tool_fn:
                    try:
                        result_text = tool_fn.invoke(tool_args)
                    except Exception as exc:
                        result_text = f"Tool error: {exc}"
                        logger.error("Tool %s failed: %s", tool_name, exc)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result_text),
                            tool_call_id=tool_call_id,
                        )
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: Unknown tool '{tool_name}'",
                            tool_call_id=tool_call_id,
                        )
                    )

        return {"messages": tool_messages, "citations": new_citations}

    return execute_tools
