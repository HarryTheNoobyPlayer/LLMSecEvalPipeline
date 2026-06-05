"""Abstract base class for code generators.

Any backend (Ollama, raw HF transformers, an API, ...) implements this
interface so the runner stays backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    """Raw output from a single generation call.

    Attributes
    ----------
    raw_response:
        The visible answer the model emitted (post-reasoning), including any
        markdown fences.  For non-reasoning models this is the full output.
        For reasoning models like DeepSeek-R1, this is what comes *after*
        the ``<think>`` block, the actual code/answer.
    thinking:
        The chain-of-thought reasoning, if the backend exposes it separately
        (Ollama splits ``<think>...</think>`` into its own ``thinking`` field
        on reasoning models).  Kept for debugging; not used by SAST.
    token_count:
        Number of tokens the model produced, if the backend reports it.
    generation_time_s:
        Wall-clock generation time in seconds, if the backend reports it.
    backend_metadata:
        Backend-specific extras (eval rates, prompt token counts, etc.).
    """

    raw_response: str
    thinking: str | None = None
    token_count: int | None = None
    generation_time_s: float | None = None
    backend_metadata: dict[str, Any] | None = None


class BaseCodeGenerator(ABC):
    """Minimal interface every code-generation backend must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Generate one response for *prompt*.

        Implementations must raise on transport / backend errors; the runner
        is responsible for catching and recording them per-prompt so a single
        failure never halts the run.
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return metadata identifying the model and its generation params.

        The returned dict is embedded into each output record so a results
        file is self-describing.
        """
