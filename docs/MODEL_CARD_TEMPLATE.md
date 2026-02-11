# BioScholar-Medical-QA

A fine-tuned Llama-3.2 model for grounded medical question answering with RAG.

## Model Details

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct (or Llama-3.2-8B)
- **Method**: QLoRA (4-bit + LoRA) / LoRA
- **Task**: Instruction-following for medical Q&A with context grounding
- **Training Data**: 50 medical Q&A pairs from clinical guidelines (gold set) + retrieved context chunks

## Intended Use

- Medical RAG systems: answer questions using only provided context
- Citation-style output with `[Source N]` notation
- **Not for direct patient diagnosis** — for research and education only

## Training

- **Data**: Gold set from `data/eval/gold_set_v1.csv` + retrieval from Qdrant
- **Format**: instruction + context + question → answer
- **LoRA**: r=64, alpha=128, target_modules: q_proj, k_proj, v_proj, o_proj
- **Epochs**: 3
- **Learning rate**: 2e-5

## Evaluation

Compared Base RAG (Ollama Llama) vs Fine-tuned RAG:

```bash
python eval/run_eval.py  # Base
python eval/run_eval.py --llm-adapter-path outputs/llm_finetune/run_xxx/final_adapter  # Fine-tuned
```

Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision, Entity Coverage, Citation Accuracy, Safety Score.

## Limitations

- Trained on 50 examples — may not generalize to all medical domains
- Requires retrieval context; does not replace full RAG pipeline
- For educational use only

## Citation

```bibtex
@misc{bioscholar-medical-qa,
  title={BioScholar Medical QA: Fine-Tuned Llama for Medical RAG},
  author={Your Name},
  year={2025},
}
```
