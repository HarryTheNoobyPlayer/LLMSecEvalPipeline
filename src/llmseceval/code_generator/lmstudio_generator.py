"""LM Studio code generator backend.

Talks to LM Studio's OpenAI-compatible HTTP API
(``/v1/chat/completions``).  LM Studio must be running with the
server enabled (default port 1234).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from llmseceval.code_generator.base import BaseCodeGenerator, GenerationResult
from llmseceval.config import CodeGeneratorConfig


class LMStudioGenerator(BaseCodeGenerator):
    """Generate code via LM Studio's OpenAI-compatible API."""

    def __init__(self, config: CodeGeneratorConfig) -> None:
        self.config = config
        self._base_url = config.ollama_host.rstrip("/")  # reuse ollama_host field
        self._model = config.model_name
        self._session = requests.Session()

    # ---- BaseCodeGenerator interface ------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        filled = self.config.prompt_template.format(prompt=prompt)
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": filled}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0

        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        # LM Studio puts the <think> block inside the content; split it out.
        thinking = None
        raw = content
        if "<think>" in content and "</think>" in content:
            think_start = content.index("<think>") + len("<think>")
            think_end = content.index("</think>")
            thinking = content[think_start:think_end].strip()
            raw = content[think_end + len("</think>"):].strip()

        return GenerationResult(
            raw_response=raw,
            thinking=thinking,
            token_count=usage.get("completion_tokens"),
            generation_time_s=round(elapsed, 3),
        )

    def get_model_info(self) -> dict:
        return {
            "backend": "lmstudio",
            "model_name": self._model,
            "base_url": self._base_url,
            "generation_params": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_new_tokens,
            },
        }
