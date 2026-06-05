"""Pull a Python code snippet out of a raw LLM response.

DeepSeek-R1 emits ``<think>...</think>`` blocks before the answer and
typically wraps code in markdown fences.  This module strips both.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_ONLY = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_BLOCK = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# Used as a fallback when generation was truncated (max_new_tokens hit)
# and the closing ``` was never emitted.
_UNCLOSED_LEADING_FENCE = re.compile(
    r"```(?:python|py)?\s*\n(.*)",
    re.DOTALL | re.IGNORECASE,
)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from *text*.

    Handles two cases:
    1. Complete ``<think>...</think>`` pairs (most common).
    2. A dangling ``</think>`` with no opening tag, some chat templates
       prepend the opener server-side, so only the closer reaches us.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" not in cleaned and "</think>" in cleaned:
        cleaned = _OPEN_THINK_ONLY.sub("", cleaned, count=1)
    return cleaned.strip()


def extract_code(raw_response: str) -> str:
    """Return the Python code from a raw model response.

    Strategy:
    1. Strip any ``<think>`` block.
    2. If at least one closed ```` ```python ```` (or unlabelled) fence
       exists, concatenate all of them (most prompts produce a single block).
    3. Else if the response *starts* with an opening fence but was truncated
       before the closer (max_new_tokens hit mid-code), strip the opener and
       use everything after it.
    4. Otherwise return the de-thought text verbatim.
    """
    cleaned = strip_think(raw_response)

    fences = _FENCED_BLOCK.findall(cleaned)
    if fences:
        return "\n\n".join(block.strip() for block in fences).strip()

    unclosed = _UNCLOSED_LEADING_FENCE.match(cleaned.lstrip())
    if unclosed:
        return unclosed.group(1).rstrip()

    return cleaned
