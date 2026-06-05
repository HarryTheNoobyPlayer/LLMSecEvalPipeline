"""Ollama HTTP backend for code generation.

Talks to an Ollama server via its REST API (``POST /api/generate``).  The
server can be local (``http://localhost:11434``) or reached through an SSH
port-forward to a remote GPU host.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from llmseceval.code_generator.base import BaseCodeGenerator, GenerationResult
from llmseceval.config import CodeGeneratorConfig

logger = logging.getLogger(__name__)


class OllamaGenerator(BaseCodeGenerator):
    """Generate code by calling Ollama's ``/api/generate`` endpoint."""

    def __init__(self, config: CodeGeneratorConfig) -> None:
        self.config = config
        self._host = config.ollama_host.rstrip("/")
        self._endpoint = f"{self._host}/api/generate"
        # A reusable session keeps the TCP connection warm across prompts.
        self._session = requests.Session()

    # ---- public API -------------------------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        """POST a single prompt to Ollama and return the structured result."""
        full_prompt = self.config.prompt_template.format(prompt=prompt)

        payload = {
            "model": self.config.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_new_tokens,
                "num_ctx": self.config.num_ctx,
                "seed": self.config.random_seed,
            },
        }

        logger.debug("POST %s model=%s", self._endpoint, self.config.model_name)
        resp = self._session.post(
            self._endpoint,
            json=payload,
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama reports total_duration in nanoseconds
        total_ns = data.get("total_duration")
        gen_time_s = total_ns / 1e9 if isinstance(total_ns, (int, float)) else None

        return GenerationResult(
            raw_response=data.get("response", ""),
            thinking=data.get("thinking"),
            token_count=data.get("eval_count"),
            generation_time_s=gen_time_s,
            backend_metadata={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
                "eval_duration_ns": data.get("eval_duration"),
                "done_reason": data.get("done_reason"),
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "ollama",
            "ollama_host": self._host,
            "model_name": self.config.model_name,
            "generation_params": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_new_tokens,
                "seed": self.config.random_seed,
            },
        }
