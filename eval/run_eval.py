#!/usr/bin/env python3
"""
BioScholar Evaluation Runner with MLflow Tracking.

Runs both RAGAS-style and custom medical metrics, then logs
everything to MLflow for experiment comparison.

Usage:
    # Run evaluation with MLflow tracking
    python eval/run_eval.py

    # Skip LLM (retrieval-only eval) — useful when Ollama isn't running
    python eval/run_eval.py --skip-llm

    # Custom gold set and experiment name
    python eval/run_eval.py --gold-set data/eval/gold_set_v1.csv --experiment bioscholar-v1

    # View results in MLflow UI
    mlflow ui --port 5000
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ragas_eval import (
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    run_pipeline,
)
from eval.medical_metrics import compute_medical_metrics
from src.inference import MedicalNER
from src.llm import LLMClient
from src.retrieve import HybridRetriever
from src.vector_store import QdrantStore


def _find_latest_model() -> str:
    """Find the most recent training run's final_model directory."""
    import glob as glob_mod
    candidates = sorted(glob_mod.glob("outputs/models/run_*/final_model"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found. Run training first.")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full evaluation with MLflow tracking")
    parser.add_argument("--gold-set", type=Path, default=Path("data/eval/gold_set_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--qdrant-path", default="data/qdrant_db")
    parser.add_argument("--collection-name", default="bio_guidelines")
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--llm-model", default="llama3.2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation")
    parser.add_argument("--experiment", default="bioscholar-eval", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name (auto-generated if None)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


def evaluate_all(
    gold_df: pd.DataFrame,
    pipeline_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all metrics (RAGAS + medical) and return a full report."""
    per_question: List[Dict[str, Any]] = []

    for i, (_, gold_row) in enumerate(gold_df.iterrows()):
        result = pipeline_results[i]

        # RAGAS metrics
        faith = compute_faithfulness(result["answer"], result["contexts"])
        relevance = compute_answer_relevance(gold_row["question"], result["answer"])
        recall = compute_context_recall(gold_row["expected_answer"], result["contexts"])
        precision = compute_context_precision(gold_row["question"], result["contexts"])

        # Medical metrics
        med = compute_medical_metrics(gold_row.to_dict(), result)

        per_question.append({
            "question": gold_row["question"],
            "category": gold_row.get("category", ""),
            "difficulty": gold_row.get("difficulty", ""),
            # RAGAS
            "faithfulness": round(faith, 4),
            "answer_relevance": round(relevance, 4),
            "context_recall": round(recall, 4),
            "context_precision": round(precision, 4),
            # Medical
            "entity_coverage": med["entity_coverage"],
            "citation_accuracy": med["citation_accuracy"],
            "safety_score": med["safety_score"],
            # Preview
            "answer_preview": result["answer"][:300],
            "num_contexts": len(result["contexts"]),
        })

    df = pd.DataFrame(per_question)

    # Aggregate metrics
    metric_cols = [
        "faithfulness", "answer_relevance", "context_recall",
        "context_precision", "entity_coverage", "citation_accuracy",
        "safety_score",
    ]
    aggregate = {col: round(df[col].mean(), 4) for col in metric_cols}

    # Per-category breakdown
    category_breakdown = {}
    if "category" in df.columns:
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            category_breakdown[cat] = {
                col: round(cat_df[col].mean(), 4) for col in metric_cols
            }
            category_breakdown[cat]["count"] = len(cat_df)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(per_question),
        "aggregate_metrics": aggregate,
        "category_breakdown": category_breakdown,
        "per_question": per_question,
    }


def log_to_mlflow(
    report: Dict[str, Any],
    config: Dict[str, Any],
    experiment_name: str,
    run_name: str,
) -> None:
    """Log evaluation results to MLflow."""
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed. Skipping tracking.")
        print("Install with: pip install mlflow")
        return

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model_path": config.get("model_path", ""),
            "collection_name": config.get("collection_name", ""),
            "top_k": config.get("top_k", 5),
            "score_threshold": config.get("score_threshold", 0.0),
            "llm_model": config.get("llm_model", ""),
            "llm_available": config.get("llm_available", False),
            "num_questions": report["num_questions"],
        })

        # Log aggregate metrics
        for name, value in report["aggregate_metrics"].items():
            mlflow.log_metric(name, value)

        # Log per-category metrics
        for cat, metrics in report.get("category_breakdown", {}).items():
            for metric_name, value in metrics.items():
                if metric_name != "count":
                    mlflow.log_metric(f"{cat}_{metric_name}", value)

        # Log the full report as an artifact
        report_path = Path("outputs") / "eval_report_latest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        print(f"MLflow run logged: {mlflow.active_run().info.run_id}")


def main() -> None:
    args = parse_args()

    # Load gold set
    if not args.gold_set.exists():
        print(f"Gold set not found: {args.gold_set}")
        print("Create it first or provide a different path.")
        sys.exit(1)

    gold_df = pd.read_csv(args.gold_set)
    print(f"Loaded {len(gold_df)} evaluation questions from {args.gold_set}")

    # Initialize components
    model_path = args.model_path or _find_latest_model()
    print(f"NER model: {model_path}")

    ner = MedicalNER(model_path=model_path)
    store = QdrantStore(
        collection_name=args.collection_name,
        qdrant_path=args.qdrant_path,
    )
    retriever = HybridRetriever(ner=ner, store=store)
    llm = LLMClient(base_url=args.llm_base_url, model=args.llm_model)

    print(f"Collection '{args.collection_name}': {store.count()} chunks")

    llm_available = llm.is_available()
    use_llm = not args.skip_llm and llm_available

    if not use_llm:
        if not args.skip_llm:
            print("\nWARNING: LLM not available — running retrieval-only evaluation.")
            print("To enable full evaluation, start Ollama: ollama serve && ollama pull llama3.2\n")
        else:
            print("\nRunning retrieval-only evaluation (--skip-llm).\n")

    # Run pipeline
    if use_llm:
        print("Running full RAG pipeline (NER + retrieval + LLM)...")
        pipeline_results = run_pipeline(
            questions=gold_df["question"].tolist(),
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )
    else:
        print("Running retrieval-only pipeline (NER + retrieval)...")
        pipeline_results = []
        for _, row in gold_df.iterrows():
            search_result = retriever.search(
                query=row["question"],
                top_k=args.top_k,
                entity_filter=True,
                score_threshold=args.score_threshold,
            )
            pipeline_results.append({
                "question": row["question"],
                "answer": row["expected_answer"],
                "contexts": [c["text"] for c in search_result["results"]],
                "source_files": [c.get("source_file", "") for c in search_result["results"]],
                "query_entities": [
                    {"text": e.text, "label": e.label, "confidence": e.confidence}
                    for e in search_result["query_entities"]
                ],
            })

    # Evaluate
    print("Computing metrics...")
    report = evaluate_all(gold_df, pipeline_results)

    # Add config to report
    config = {
        "model_path": model_path,
        "collection_name": args.collection_name,
        "qdrant_path": args.qdrant_path,
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "llm_model": args.llm_model,
        "llm_available": llm_available,
        "skip_llm": not use_llm,
    }
    report["config"] = config

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"eval_report_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # MLflow logging
    if not args.no_mlflow:
        run_name = args.run_name or f"eval_{timestamp}"
        log_to_mlflow(report, config, args.experiment, run_name)

    # Print summary
    agg = report["aggregate_metrics"]
    print("\n" + "=" * 60)
    print("BioScholar Evaluation Report")
    print("=" * 60)
    print("\n  RAGAS Metrics:")
    print(f"    Faithfulness:       {agg['faithfulness']:.4f}")
    print(f"    Answer Relevance:   {agg['answer_relevance']:.4f}")
    print(f"    Context Recall:     {agg['context_recall']:.4f}")
    print(f"    Context Precision:  {agg['context_precision']:.4f}")
    print("\n  Medical Metrics:")
    print(f"    Entity Coverage:    {agg['entity_coverage']:.4f}")
    print(f"    Citation Accuracy:  {agg['citation_accuracy']:.4f}")
    print(f"    Safety Score:       {agg['safety_score']:.4f}")

    if report.get("category_breakdown"):
        print("\n  Per-Category Breakdown:")
        for cat, metrics in report["category_breakdown"].items():
            print(f"    {cat} (n={metrics['count']}):")
            print(f"      Context Recall: {metrics['context_recall']:.4f}  "
                  f"Entity Coverage: {metrics['entity_coverage']:.4f}")

    print(f"\n  Questions evaluated: {report['num_questions']}")
    print(f"  Mode: {'Full RAG' if use_llm else 'Retrieval-only'}")
    print(f"  Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
