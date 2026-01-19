# Symptom NER Fine-Tuning Pipeline

Fine-tune BioBERT with LoRA for medical Named Entity Recognition (NER) - extracting diseases, chemicals, and symptoms from biomedical text.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
python data/prepare_data.py

# 4. Train model
python src/train.py

# 5. Run inference
python src/predict.py --text "I have persistent headache and fever for 3 days"
```

## Project Structure

```
medai/
├── config.yaml              # Hyperparameters and model settings
├── requirements.txt         # Dependencies
├── data/
│   ├── prepare_data.py      # Download and preprocess datasets
│   ├── synthetic_symptoms.csv  # Custom symptom examples
│   └── processed/           # Cached processed datasets
├── notebooks/
│   └── 01_explore_pretrained.ipynb  # Baseline exploration
├── src/
│   ├── dataset.py           # NER Dataset class
│   ├── model.py             # BioBERT + LoRA model
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation metrics
│   └── predict.py           # CLI inference
├── tests/
│   ├── test_dataset.py
│   └── test_predict.py
└── outputs/
    ├── models/              # Saved LoRA adapters
    └── logs/                # Training logs
```

## Dataset

- **BC5CDR**: 1,500 PubMed abstracts with Chemical and Disease annotations
- **Synthetic**: 50-100 custom symptom examples for evaluation

## Model

- **Base**: BioBERT (`dmis-lab/biobert-base-cased-v1.2`)
- **Fine-tuning**: LoRA (r=16, alpha=32) for parameter-efficient training
- **Task**: Token classification with BIO tagging

## Hardware Requirements

- **Minimum**: 8GB RAM, any CPU
- **Recommended**: M1/M2 Mac (MPS) or CUDA GPU
- **Training time**: ~15-30 min on M2 Pro for 3 epochs

## Expected Results

| Metric | Target | Typical |
|--------|--------|---------|
| F1 Score | >0.75 | 0.78-0.85 |
| Precision | >0.70 | 0.75-0.85 |
| Recall | >0.70 | 0.75-0.85 |

## Example Output

```json
{
  "text": "I have persistent headache and fever for 3 days",
  "entities": [
    {"text": "headache", "label": "SYMPTOM", "confidence": 0.92, "span": [21, 29]},
    {"text": "fever", "label": "SYMPTOM", "confidence": 0.88, "span": [34, 39]}
  ]
}
```

## License

MIT License - for educational purposes only.
