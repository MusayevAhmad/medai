"""
Local LLM Client for Fine-Tuned Models

Loads a HuggingFace PEFT/LoRA model for inference (no Ollama needed).
Used when evaluating the fine-tuned RAG system.

Usage:
    from src.llm_local import LocalLLMClient

    client = LocalLLMClient(adapter_path="outputs/llm_finetune/run_xxx/final_adapter")
    answer = client.generate_answer(question="...", context_chunks=[...])
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Set project-local HF cache early (before any HF imports)
_project_root = Path(__file__).parent.parent
_hf_cache = _project_root / "data" / "hf_cache"
_hf_cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf_cache)
os.environ["HF_HUB_CACHE"] = str(_hf_cache / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(_hf_cache / "hub")

# Lazy imports to avoid loading heavy deps when not used
_transformer_modules = None


def _get_transformers():
    global _transformer_modules
    if _transformer_modules is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        _transformer_modules = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "PeftModel": PeftModel,
        }
    return _transformer_modules


def _format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into numbered context."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        parts.append(f"[Source {i}] ({source}, p.{page})\n{text}")
    return "\n\n".join(parts)


class LocalLLMClient:
    """
    LLM client that loads a fine-tuned PEFT model from disk.

    Same interface as src.llm.LLMClient for drop-in replacement in eval.
    """

    _SYSTEM_PROMPT = """\
You are a medical research assistant. Answer the user's question using ONLY \
the context provided below. Follow these rules strictly:

1. Base your answer entirely on the provided context. Do not use outside knowledge.
2. Cite your sources using [Source N] notation (e.g., [Source 1], [Source 2]).
3. If the context does not contain enough information to answer the question, \
   say "I don't have enough information to answer this question."
4. Be concise but thorough. Use medical terminology accurately.
5. If multiple sources support a claim, cite all of them."""

    _CONTEXT_TEMPLATE = """\
--- Context ---
{context}
--- End Context ---

Question: {question}"""

    def __init__(
        self,
        adapter_path: str,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """
        Args:
            adapter_path: Path to saved PEFT adapter (final_adapter dir)
            base_model: Base model ID (auto-detected from adapter_config.json if None)
            device: Device to run on (auto-detect if None)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.adapter_path = Path(adapter_path)
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # Load base model from adapter config if needed
        config_path = self.adapter_path / "adapter_config.json"
        if base_model is None and config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct")
        self.base_model = base_model or "meta-llama/Llama-3.2-3B-Instruct"

        import torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._tokenizer = None
        self.model = f"local:{adapter_path}"  # For logging/compatibility

    def _load_model(self):
        if self._model is not None:
            return
        mods = _get_transformers()
        AutoModelForCausalLM = mods["AutoModelForCausalLM"]
        AutoTokenizer = mods["AutoTokenizer"]
        PeftModel = mods["PeftModel"]

        logger.info("Loading base model: %s", self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            trust_remote_code=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, self.adapter_path)
        if self.device != "cuda":
            model = model.to(self.device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        logger.info("Local LLM loaded successfully")

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer from context + question (same interface as LLMClient)."""
        if not context_chunks:
            return "I don't have any relevant information to answer this question."

        self._load_model()
        context_str = _format_context(context_chunks)
        user_msg = self._CONTEXT_TEMPLATE.format(
            context=context_str,
            question=question,
        )
        system = system_prompt or self._SYSTEM_PROMPT

        # Chat format for Llama 3.2
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True)
        return answer.strip()

    def is_available(self) -> bool:
        """Always True once adapter path exists."""
        return self.adapter_path.exists()
