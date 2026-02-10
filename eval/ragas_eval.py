#!/usr/bin/env python3
"""
RAGAS Evaluation for BioScholar RAG Pipeline.

Evaluates the RAG system using the RAGAS framework metrics:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevance: Does the answer address the question?
    - Context Recall: Did we retrieve the right context?
    - Context Precision: Are the retrieved contexts relevant?

Usage:
    python eval/ragas_eval.py
    python eval/ragas_eval.py --gold-set data/eval/gold_set_v1.csv --output outputs/eval_report_v1.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import MedicalNER
from src.llm import LLMClient
from src.retrieve import HybridRetriever
from src.vector_store import QdrantStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runner — get system answers for the gold set
# ---------------------------------------------------------------------------

def run_pipeline(
    questions: List[str],
    retriever: HybridRetriever,
    llm: LLMClient,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run the full RAG pipeline on a list of questions.

    Returns a list of dicts with keys:
        question, answer, contexts, source_files
    """
    results: List[Dict[str, Any]] = []

    for question in questions:
        search_result = retriever.search(
            query=question,
            top_k=top_k,
            entity_filter=True,
            score_threshold=score_threshold,
        )

        chunks = search_result["results"]
        context_texts = [c["text"] for c in chunks]
        source_files = [c.get("source_file", "") for c in chunks]

        if not chunks:
            answer = "I don't have enough relevant information to answer this question."
        else:
            try:
                answer = llm.generate_answer(
                    question=question,
                    context_chunks=chunks,
                )
            except (ConnectionError, RuntimeError) as e:
                answer = f"[LLM Error] {e}"

        results.append({
            "question": question,
            "answer": answer,
            "contexts": context_texts,
            "source_files": source_files,
            "query_entities": [
                {"text": e.text, "label": e.label, "confidence": e.confidence}
                for e in search_result["query_entities"]
            ],
        })

    return results


# ---------------------------------------------------------------------------
# RAGAS-style metrics (standalone, no ragas library dependency)
# ---------------------------------------------------------------------------

def _sentence_overlap(text_a: str, text_b: str) -> float:
    """Simple word-overlap ratio between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """Faithfulness: fraction of answer sentences supported by context.

    A simple heuristic version that checks word overlap between each
    answer sentence and the concatenated context. For production use,
    replace with LLM-based NLI or the ragas library.
    """
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    if not sentences:
        return 0.0

    full_context = " ".join(contexts).lower()
    supported = 0
    for sentence in sentences:
        overlap = _sentence_overlap(sentence, full_context)
        if overlap > 0.15:  # threshold: at least 15% word overlap
            supported += 1

    return supported / len(sentences)


def compute_answer_relevance(question: str, answer: str) -> float:
    """Answer Relevance: word overlap between question and answer.

    A lightweight proxy. For production, use embedding similarity or
    LLM-based scoring.
    """
    return _sentence_overlap(question, answer)


def compute_context_recall(
    expected_answer: str,
    contexts: List[str],
) -> float:
    """Context Recall: how much of the expected answer is covered by contexts.

    Measures the fraction of expected-answer tokens found in retrieved context.
    """
    expected_words = set(expected_answer.lower().split())
    context_words = set(" ".join(contexts).lower().split())
    if not expected_words:
        return 0.0
    recalled = expected_words & context_words
    return len(recalled) / len(expected_words)


def compute_context_precision(
    question: str,
    contexts: List[str],
) -> float:
    """Context Precision: average relevance of retrieved contexts to the question."""
    if not contexts:
        return 0.0
    scores = [_sentence_overlap(question, ctx) for ctx in contexts]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Evaluation orchestrator
# ---------------------------------------------------------------------------

def evaluate(
    gold_df: pd.DataFrame,
    pipeline_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all RAGAS-style metrics and return a report dict."""

    rows: List[Dict[str, Any]] = []

    for i, (_, gold_row) in enumerate(gold_df.iterrows()):
        result = pipeline_results[i]

        faith = compute_faithfulness(result["answer"], result["contexts"])
        relevance = compute_answer_relevance(gold_row["question"], result["answer"])
        recall = compute_context_recall(gold_row["expected_answer"], result["contexts"])
        precision = compute_context_precision(gold_row["question"], result["contexts"])

        rows.append({
            "question": gold_row["question"],
            "category": gold_row.get("category", ""),
            "difficulty": gold_row.get("difficulty", ""),
            "faithfulness": round(faith, 4),
            "answer_relevance": round(relevance, 4),
            "context_recall": round(recall, 4),
            "context_precision": round(precision, 4),
            "answer_preview": result["answer"][:200],
        })

    # Aggregate
    metrics_df = pd.DataFrame(rows)
    aggregate = {
        "faithfulness": round(metrics_df["faithfulness"].mean(), 4),
        "answer_relevance": round(metrics_df["answer_relevance"].mean(), 4),
        "context_recall": round(metrics_df["context_recall"].mean(), 4),
        "context_precision": round(metrics_df["context_precision"].mean(), 4),
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(rows),
        "aggregate_metrics": aggregate,
        "per_question": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on gold set")
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=Path("data/eval/gold_set_v1.csv"),
        help="Path to gold set CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval_report_v1.json"),
        help="Path to write the evaluation report",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="NER model path (auto-detects latest if None)",
    )
    parser.add_argument(
        "--qdrant-path",
        default="data/qdrant_db",
        help="Local Qdrant storage path",
    )
    parser.add_argument(
        "--collection-name",
        default="bio_guidelines",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:11434/v1",
        help="LLM API base URL",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2",
        help="LLM model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation (evaluate retrieval only)",
    )
    return parser.parse_args()


def _find_latest_model() -> str:
    """Find the most recent training run's final_model directory."""
    import glob as glob_mod
    candidates = sorted(glob_mod.glob("outputs/models/run_*/final_model"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found. Run training first.")
    return candidates[0]


def main() -> None:
    args = parse_args()

    # Load gold set
    if not args.gold_set.exists():
        print(f"Gold set not found: {args.gold_set}")
        sys.exit(1)

    gold_df = pd.read_csv(args.gold_set)
    print(f"Loaded {len(gold_df)} questions from {args.gold_set}")

    # Initialize components
    model_path = args.model_path or _find_latest_model()
    print(f"Using NER model: {model_path}")

    ner = MedicalNER(model_path=model_path)
    store = QdrantStore(
        collection_name=args.collection_name,
        qdrant_path=args.qdrant_path,
    )
    retriever = HybridRetriever(ner=ner, store=store)

    llm = LLMClient(base_url=args.llm_base_url, model=args.llm_model)

    print(f"Qdrant collection '{args.collection_name}': {store.count()} chunks")
    print(f"LLM available: {llm.is_available()}")

    if args.skip_llm or not llm.is_available():
        if not args.skip_llm:
            print("WARNING: LLM not available. Running retrieval-only evaluation.")
            print("Start Ollama with: ollama serve && ollama pull llama3.2")

        # Retrieval-only mode: just test context retrieval quality
        pipeline_results = []
        for _, row in gold_df.iterrows():
            search_result = retriever.search(
                query=row["question"],
                top_k=args.top_k,
                entity_filter=True,
            )
            pipeline_results.append({
                "question": row["question"],
                "answer": row["expected_answer"],  # use expected as answer placeholder
                "contexts": [c["text"] for c in search_result["results"]],
                "source_files": [c.get("source_file", "") for c in search_result["results"]],
                "query_entities": [
                    {"text": e.text, "label": e.label, "confidence": e.confidence}
                    for e in search_result["query_entities"]
                ],
            })
    else:
        # Full RAG evaluation
        print("\nRunning full RAG pipeline...")
        pipeline_results = run_pipeline(
            questions=gold_df["question"].tolist(),
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
        )

    # Evaluate
    print("\nComputing metrics...")
    report = evaluate(gold_df, pipeline_results)

    # Add metadata
    report["config"] = {
        "model_path": model_path,
        "collection_name": args.collection_name,
        "top_k": args.top_k,
        "llm_model": args.llm_model,
        "llm_available": llm.is_available(),
        "skip_llm": args.skip_llm or not llm.is_available(),
    }

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Report")
    print("=" * 60)
    agg = report["aggregate_metrics"]
    print(f"  Faithfulness:      {agg['faithfulness']:.4f}")
    print(f"  Answer Relevance:  {agg['answer_relevance']:.4f}")
    print(f"  Context Recall:    {agg['context_recall']:.4f}")
    print(f"  Context Precision: {agg['context_precision']:.4f}")
    print(f"\nQuestions evaluated: {report['num_questions']}")
    print(f"Report saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
