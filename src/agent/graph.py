"""LangGraph workflow for multi-step medical reasoning.

Graph shape (ReAct loop):
    agent -> tools -> agent -> ... -> END
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.agent.tools import build_tools

logger = logging.getLogger(__name__)

MAX_STEPS = 6


def is_complex_query(question: str, entities: Sequence[Any]) -> bool:
    """Heuristic complexity detection used by ``/query`` auto-routing."""
    q = question.lower()
    patterns = [
        r"\bcompare\b",
        r"\bversus\b|\bvs\b",
        r"\bpros and cons\b",
        r"\bside effects?\b",
        r"\binteraction\b",
        r"\bhow do .* differ\b",
        r"\band\b.*\bhow\b",
    ]
    return any(re.search(p, q) for p in patterns) or len(entities) >= 2


def _agent_prompt() -> str:
    return (
        "You are a medical evidence assistant. Use tools for retrieval and factual checks. "
        "When comparing drugs, call lookup_drug_interaction if relevant. "
        "Always ground claims in tool results and provide concise final answers."
    )


def _safe_tool_args(args: Any) -> Dict[str, Any]:
    """Normalize tool args from dict or small-model malformed JSON strings."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"query": text}
    return {}


def build_agent_graph(llm: Any, retriever: Any) -> Any:
    """Compile and return the LangGraph state machine."""
    tools = build_tools(retriever)

    tool_specs = [
        StructuredTool.from_function(
            func=tools["search_guidelines"],
            name="search_guidelines",
            description="Search indexed guideline evidence by query.",
        ),
        StructuredTool.from_function(
            func=tools["lookup_drug_interaction"],
            name="lookup_drug_interaction",
            description="Look up known interaction data for two drugs.",
        ),
        StructuredTool.from_function(
            func=tools["summarize_section"],
            name="summarize_section",
            description="Summarize a section from a specific source file.",
        ),
    ]

    llm_with_tools = llm.bind_tools(tool_specs)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content=state["question"])]

        prompt = HumanMessage(content=_agent_prompt())
        response = llm_with_tools.invoke([prompt] + list(messages))
        return {"messages": [response]}

    def tools_node(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {"messages": []}

        out_messages: List[BaseMessage] = []
        out_citations: List[Dict[str, Any]] = []

        for call in last.tool_calls or []:
            name = call.get("name")
            raw_args = call.get("args", {})
            args = _safe_tool_args(raw_args)
            fn = tools.get(name)
            if fn is None:
                result = {"error": f"Unknown tool: {name}", "citations": []}
            else:
                try:
                    result = fn(**args)
                except TypeError:
                    if name == "search_guidelines" and "query" not in args and isinstance(raw_args, str):
                        result = fn(query=raw_args)
                    else:
                        result = {"error": f"Invalid arguments for {name}", "citations": []}

            citations = result.get("citations", []) if isinstance(result, dict) else []
            out_citations.extend(citations)
            out_messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call.get("id", name or "tool"),
                )
            )

        return {"messages": out_messages, "citations": out_citations}

    def route(state: AgentState) -> str:
        steps = sum(isinstance(m, AIMessage) for m in state.get("messages", []))
        if steps >= MAX_STEPS:
            return END

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(graph: Any, question: str) -> Dict[str, Any]:
    """Run the compiled graph and return normalized output."""
    initial = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "citations": [],
    }
    result = graph.invoke(initial)
    messages = result.get("messages", [])

    final_answer = "I don't have enough information to answer this question."
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            final_answer = str(message.content)
            break

    steps = sum(isinstance(m, AIMessage) for m in messages)
    return {
        "answer": final_answer,
        "steps": steps,
        "citations": result.get("citations", []),
    }
