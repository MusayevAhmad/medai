"""
LLM Client for answer generation.

Wraps an OpenAI-compatible API (works with Ollama, OpenAI, Anthropic, etc.)
for grounded medical question answering with citations.

Usage:
    from src.llm import LLMClient

    client = LLMClient()  # defaults to Ollama at localhost:11434
    answer = client.generate_answer(question="What causes fever?", context_chunks=[...])
"""

import logging
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default prompt template for grounded medical QA
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


def _format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context string.

    Args:
        chunks: List of search result dicts with 'text', 'source_file',
                'page_number' keys.

    Returns:
        Formatted context string with source labels.
    """
    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        parts.append(f"[Source {i}] ({source}, p.{page})\n{text}")
    return "\n\n".join(parts)


class LLMClient:
    """Client for OpenAI-compatible LLM APIs.

    Supports Ollama (default), OpenAI, and any provider that implements
    the ``/v1/chat/completions`` endpoint.

    Args:
        base_url: API base URL. Defaults to Ollama's local server.
        model: Model name to use.
        api_key: API key (not needed for Ollama).
        timeout: Request timeout in seconds.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the response.

    Example:
        >>> client = LLMClient()
        >>> client.generate_answer("What causes fever?", context_chunks)
        "Fever is caused by..."
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        api_key: str = "ollama",
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a grounded answer using retrieved context.

        Args:
            question: The user's medical question.
            context_chunks: Retrieved chunks from the vector store.
            system_prompt: Custom system prompt (uses default if None).

        Returns:
            The generated answer text.

        Raises:
            ConnectionError: If the LLM server is unreachable.
            RuntimeError: If the API returns an error.
        """
        if not context_chunks:
            return "I don't have any relevant information to answer this question."

        system = system_prompt or _SYSTEM_PROMPT
        context_str = _format_context(context_chunks)
        user_message = _CONTEXT_TEMPLATE.format(
            context=context_str,
            question=question,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                "If using Ollama, make sure it is running: `ollama serve`. "
                f"Details: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM API error ({e.response.status_code}): {e.response.text}"
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        """Check if the LLM server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/models",
                                  headers={"Authorization": f"Bearer {self.api_key}"})
                return resp.status_code == 200
        except Exception:
            return False
