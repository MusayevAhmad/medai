"""
BioScholar LangGraph Agent

Multi-step reasoning agent for complex medical questions.  Uses a ReAct
pattern with tool-calling to decompose and answer comparison queries,
multi-part questions, and other complex information needs.

Usage::

    from src.agent import build_agent_graph, run_agent, is_complex_query

    graph = build_agent_graph(llm=chat_model, retriever=retriever)
    result = run_agent(graph, "Compare aspirin and ibuprofen for pain relief")
    print(result["answer"])
"""

from src.agent.graph import build_agent_graph, is_complex_query, run_agent

__all__ = ["build_agent_graph", "is_complex_query", "run_agent"]
