#!/usr/bin/env python3
"""
Phase 7.1: Generate Training Data from Retrieved Chunks

Builds instruction-tuning data for medical Q&A by:
1. Loading the gold set (question, expected_answer pairs)
2. Running retrieval for each question to get context chunks
3. Formatting as (instruction, input, output) for SFT/QLoRA training

Output: JSONL file compatible with TRL SFTTrainer / HuggingFace datasets.

Usage:
    python scripts/generate_finetune_data.py
    python scripts/generate_finetune_data.py --output data/finetune/medqa_train.jsonl
    python scripts/generate_finetune_data.py --top-k 8 --augment
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import MedicalNER
from src.retrieve import HybridRetriever
from src.vector_store import QdrantStore


def _find_latest_ner_model() -> str:
    """Find the most recent NER model directory."""
    import glob as glob_mod
    candidates = sorted(
        glob_mod.glob("outputs/models/run_*/final_model"), reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            "No trained NER model found. Run training first: python src/train.py"
        )
    return candidates[0]


def _format_context(chunks: list) -> str:
    """Format chunks into context string for the model."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        source = c.get("source_file", "unknown")
        page = c.get("page_number", "?")
        text = c.get("text", "")
        parts.append(f"[Source {i}] ({source}, p.{page})\n{text}")
    return "\n\n".join(parts)


def generate_finetune_data(
    gold_set_path: Path,
    output_path: Path,
    model_path: str,
    qdrant_path: str,
    collection_name: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    augment_synthetic: bool = False,
) -> int:
    """
    Generate instruction-tuning data from gold set + retrieval.

    Each example is formatted as:
    - instruction: "Answer the following medical question using only the provided context."
    - input: "Context: ...\nQuestion: ..."
    - output: The expected/generated answer with citations

    Args:
        gold_set_path: Path to gold_set_v1.csv
        output_path: Path to output JSONL file
        model_path: Path to NER model
        qdrant_path: Path to Qdrant database
        collection_name: Qdrant collection name
        top_k: Number of chunks to retrieve per question
        score_threshold: Min similarity score for chunks
        augment_synthetic: If True, add synthetic Q&A from chunks (optional)

    Returns:
        Number of examples written
    """
    gold_df = pd.read_csv(gold_set_path)
    print(f"Loaded {len(gold_df)} questions from {gold_set_path}")

    # Initialize retrieval pipeline
    ner = MedicalNER(model_path=model_path)
    store = QdrantStore(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )
    retriever = HybridRetriever(ner=ner, store=store)
    print(f"Collection '{collection_name}': {store.count()} chunks")

    # Alpaca-style format for instruction tuning
    examples = []

    for idx, row in gold_df.iterrows():
        question = row["question"].strip()
        expected_answer = row["expected_answer"].strip()

        # Retrieve context
        search_result = retriever.search(
            query=question,
            top_k=top_k,
            entity_filter=True,
            score_threshold=score_threshold,
        )
        chunks = search_result["results"]

        # Format context
        context_str = _format_context(chunks) if chunks else "No relevant context found."

        # Build instruction-tuning example
        # Format matches our RAG prompt: context + question -> answer with [Source N]
        instruction = (
            "You are a medical research assistant. Answer the user's question "
            "using ONLY the context provided. Base your answer entirely on the "
            "context. Cite sources using [Source N] notation. If the context "
            "does not contain enough information, say so."
        )
        input_text = f"--- Context ---\n{context_str}\n--- End Context ---\n\nQuestion: {question}"
        output_text = expected_answer

        examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "question": question,
            "category": row.get("category", ""),
            "difficulty": row.get("difficulty", ""),
            "num_chunks": len(chunks),
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(gold_df)} questions")

    # Optional: augment with synthetic examples from chunks
    if augment_synthetic and store.count() > 0:
        # Sample chunks and create simple Q&A (e.g., "What does this section say?")
        # For now, we skip complex augmentation - can be extended later
        print("  (Augment synthetic: placeholder - extend as needed)")

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")
    return len(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning data from gold set + retrieval"
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=Path("data/eval/gold_set_v1.csv"),
        help="Path to gold set CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finetune/medqa_train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="NER model path (auto-detects latest if not set)",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="data/qdrant_db",
        help="Qdrant database path",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="bio_guidelines",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Chunks to retrieve per question",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score for chunks",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Add synthetic examples (placeholder)",
    )
    args = parser.parse_args()

    if not args.gold_set.exists():
        print(f"Gold set not found: {args.gold_set}")
        sys.exit(1)

    model_path = args.model_path or _find_latest_ner_model()
    print(f"NER model: {model_path}")

    count = generate_finetune_data(
        gold_set_path=args.gold_set,
        output_path=args.output,
        model_path=model_path,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        augment_synthetic=args.augment,
    )

    print(f"\nDone. {count} training examples saved to {args.output}")


if __name__ == "__main__":
    main()
