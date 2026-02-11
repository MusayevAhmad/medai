#!/usr/bin/env python3
"""
Quick smoke test for the LangGraph agent (Task 5.1).

Tests:
    1. Simple question (should NOT trigger agent in auto mode)
    2. Comparison question (should trigger agent)
    3. Direct agent invocation

Usage:
    python scripts/test_agent.py
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("test_agent")


def main():
    # ------------------------------------------------------------------
    # 1. Initialise dependencies (same as FastAPI startup)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BioScholar Agent Smoke Test (Task 5.1)")
    logger.info("=" * 60)

    logger.info("Loading NER model + vector store + retriever...")
    from app.dependencies import init_dependencies, get_agent_graph, get_retriever

    init_dependencies(
        collection_name="bio_guidelines",
        qdrant_path="data/qdrant_db",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.2",
    )

    agent_graph = get_agent_graph()
    if agent_graph is None:
        logger.error("Agent graph failed to initialise!")
        sys.exit(1)

    logger.info("Agent graph ready!")

    # ------------------------------------------------------------------
    # 2. Test complexity detection
    # ------------------------------------------------------------------
    from src.agent.graph import is_complex_query

    test_cases = [
        ("What causes fever?", False),
        ("Compare aspirin and ibuprofen for pain relief", True),
        ("What is the difference between Type 1 and Type 2 diabetes?", True),
        ("Describe hypertension treatment", False),
    ]

    logger.info("\n--- Complexity Detection Tests ---")
    for question, expected in test_cases:
        result = is_complex_query(question)
        status = "PASS" if result == expected else "FAIL"
        logger.info("  [%s] %r -> %s (expected %s)", status, question, result, expected)

    # ------------------------------------------------------------------
    # 3. Run the agent on a comparison question
    # ------------------------------------------------------------------
    from src.agent.graph import run_agent

    test_question = "Compare the treatment approaches for hypertension and diabetes"

    logger.info("\n--- Agent Execution Test ---")
    logger.info("Question: %s", test_question)
    logger.info("Running agent...")

    start = time.time()
    result = run_agent(agent_graph, test_question)
    elapsed = time.time() - start

    logger.info("Agent completed in %.1f seconds", elapsed)
    logger.info("Steps: %d", result["steps"])
    logger.info("Citations: %d", len(result["citations"]))
    logger.info("\n--- Answer ---")
    print(result["answer"])

    if result["citations"]:
        logger.info("\n--- Citations ---")
        for i, c in enumerate(result["citations"][:5], 1):
            logger.info(
                "  [%d] %s (p.%s, score=%.3f)",
                i,
                c["source_file"],
                c["page_number"],
                c["score"],
            )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    logger.info("\n--- Summary ---")
    logger.info("Agent steps: %d", result["steps"])
    logger.info("Unique citations: %d", len(result["citations"]))
    logger.info("Answer length: %d chars", len(result["answer"]))
    logger.info("Time: %.1fs", elapsed)

    if result["answer"] and len(result["answer"]) > 50:
        logger.info("SMOKE TEST PASSED!")
    else:
        logger.warning("Answer seems too short — check agent behaviour")


if __name__ == "__main__":
    main()
