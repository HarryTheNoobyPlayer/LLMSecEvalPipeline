"""Unit tests for the code generator stage.

Covers:
- ``extract_code`` / ``strip_think`` edge cases
- ``OllamaGenerator`` request shaping & response parsing (with mocked HTTP)
- ``CodeGeneratorRunner`` checkpoint / resume / per-prompt error isolation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llmseceval.code_generator import (
    CodeGeneratorRunner,
    GenerationResult,
    OllamaGenerator,
    extract_code,
    strip_think,
)
from llmseceval.config import CodeGeneratorConfig


# ===========================================================================
# extractor
# ===========================================================================


class TestStripThink:
    def test_removes_complete_think_block(self) -> None:
        raw = "<think>I should write a function.</think>\ndef foo(): pass"
        assert strip_think(raw) == "def foo(): pass"

    def test_removes_multiline_think_block(self) -> None:
        raw = "<think>\nStep 1.\nStep 2.\n</think>\nprint('hi')"
        assert strip_think(raw) == "print('hi')"

    def test_handles_dangling_closing_tag(self) -> None:
        # Some chat templates inject <think> server-side; only the closer arrives.
        raw = "Reasoning text here.</think>\ndef foo(): pass"
        assert strip_think(raw) == "def foo(): pass"

    def test_no_think_block_returns_input(self) -> None:
        assert strip_think("def foo(): pass") == "def foo(): pass"

    def test_strips_surrounding_whitespace(self) -> None:
        assert strip_think("   \nhello\n  ") == "hello"


class TestExtractCode:
    def test_extracts_python_fenced_block(self) -> None:
        raw = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        assert extract_code(raw) == "def add(a, b):\n    return a + b"

    def test_extracts_unlabelled_fence(self) -> None:
        raw = "```\nprint('x')\n```"
        assert extract_code(raw) == "print('x')"

    def test_strips_think_before_fence(self) -> None:
        raw = "<think>plan</think>\n```python\nx = 1\n```"
        assert extract_code(raw) == "x = 1"

    def test_concatenates_multiple_fences(self) -> None:
        raw = "```python\nimport os\n```\nThen:\n```python\nprint('ok')\n```"
        assert extract_code(raw) == "import os\n\nprint('ok')"

    def test_falls_back_to_raw_when_no_fence(self) -> None:
        raw = "<think>...</think>\ndef bare(): pass"
        assert extract_code(raw) == "def bare(): pass"

    def test_unclosed_leading_fence_is_stripped(self) -> None:
        # Truncated generation: opening ```python but max_new_tokens cut off
        # before the closing ```.
        raw = "```python\ndef partial():\n    return 1"
        assert extract_code(raw) == "def partial():\n    return 1"

    def test_unclosed_leading_fence_after_think(self) -> None:
        raw = "<think>plan</think>\n```python\nimport os\n# truncated here"
        assert extract_code(raw) == "import os\n# truncated here"

    def test_empty_input(self) -> None:
        assert extract_code("") == ""


# ===========================================================================
# OllamaGenerator
# ===========================================================================


def _make_config(**overrides: Any) -> CodeGeneratorConfig:
    base = dict(
        backend="ollama",
        ollama_host="http://localhost:11434",
        model_name="test-model:latest",
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=128,
        num_ctx=2048,
        timeout_s=10,
        prompt_template="Task: {prompt}",
        random_seed=42,
    )
    base.update(overrides)
    return CodeGeneratorConfig(**base)


class TestOllamaGenerator:
    def test_generate_posts_expected_payload(self) -> None:
        cfg = _make_config()
        gen = OllamaGenerator(cfg)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "def foo(): pass",
            "thinking": "I should write a function named foo.",
            "eval_count": 12,
            "total_duration": 1_500_000_000,
            "done_reason": "stop",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(gen._session, "post", return_value=mock_resp) as mock_post:
            result = gen.generate("write a function")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert mock_post.call_args.args[0] == "http://localhost:11434/api/generate"
        payload = call_kwargs["json"]
        assert payload["model"] == "test-model:latest"
        assert payload["prompt"] == "Task: write a function"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.6
        assert payload["options"]["top_p"] == 0.95
        assert payload["options"]["num_predict"] == 128
        assert payload["options"]["num_ctx"] == 2048
        assert payload["options"]["seed"] == 42
        assert call_kwargs["timeout"] == 10

        assert isinstance(result, GenerationResult)
        assert result.raw_response == "def foo(): pass"
        assert result.thinking == "I should write a function named foo."
        assert result.token_count == 12
        assert result.generation_time_s == pytest.approx(1.5)

    def test_generate_handles_missing_thinking_field(self) -> None:
        # Non-reasoning models won't return a `thinking` field at all.
        cfg = _make_config()
        gen = OllamaGenerator(cfg)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "print('hi')",
            "eval_count": 5,
            "total_duration": 500_000_000,
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(gen._session, "post", return_value=mock_resp):
            result = gen.generate("anything")
        assert result.thinking is None
        assert result.raw_response == "print('hi')"

    def test_strips_trailing_slash_from_host(self) -> None:
        cfg = _make_config(ollama_host="http://localhost:11434/")
        gen = OllamaGenerator(cfg)
        assert gen._endpoint == "http://localhost:11434/api/generate"

    def test_get_model_info(self) -> None:
        cfg = _make_config()
        info = OllamaGenerator(cfg).get_model_info()
        assert info["backend"] == "ollama"
        assert info["model_name"] == "test-model:latest"
        assert info["generation_params"]["temperature"] == 0.6
        assert info["generation_params"]["seed"] == 42

    def test_raises_on_http_error(self) -> None:
        import requests

        cfg = _make_config()
        gen = OllamaGenerator(cfg)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(gen._session, "post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                gen.generate("anything")


# ===========================================================================
# CodeGeneratorRunner
# ===========================================================================


class _FakeGenerator:
    """Backend that returns canned responses keyed by prompt text."""

    def __init__(self, responses: dict[str, str], failures: set[str] | None = None) -> None:
        self.responses = responses
        self.failures = failures or set()
        self.calls: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.calls.append(prompt)
        if prompt in self.failures:
            raise RuntimeError(f"simulated failure for: {prompt}")
        return GenerationResult(
            raw_response=self.responses.get(prompt, "```python\npass\n```"),
            thinking="fake reasoning",
            token_count=5,
            generation_time_s=0.1,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "model_name": "fake:v1",
            "generation_params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 64, "seed": 1},
        }


def _write_prompts(path: Path, prompts: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for pid, text in prompts:
            fh.write(json.dumps({"prompt_id": pid, "text": text}) + "\n")


class TestCodeGeneratorRunner:
    def test_happy_path_writes_jsonl_and_py_files(self, tmp_path: Path) -> None:
        prompts_path = tmp_path / "prompts.jsonl"
        _write_prompts(prompts_path, [("a1", "write hello world"), ("b2", "add two numbers")])

        gen = _FakeGenerator(
            responses={
                "write hello world": "<think>plan</think>\n```python\nprint('hi')\n```",
                "add two numbers": "```python\ndef add(a,b): return a+b\n```",
            }
        )
        runner = CodeGeneratorRunner(gen, _make_config())
        out_dir = tmp_path / "out"
        out_jsonl = runner.run(prompts_path, out_dir)

        lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        recs = [json.loads(line) for line in lines]
        assert {r["prompt_id"] for r in recs} == {"a1", "b2"}
        rec_a = next(r for r in recs if r["prompt_id"] == "a1")
        assert rec_a["generated_code"] == "print('hi')"
        assert rec_a["raw_response"].startswith("<think>")
        assert rec_a["thinking"] == "fake reasoning"
        assert rec_a["error"] is None

        assert (out_dir / "code_files" / "a1.py").read_text(encoding="utf-8") == "print('hi')"
        assert (out_dir / "code_files" / "b2.py").exists()

        report = json.loads((out_dir / "code_generator_report.json").read_text(encoding="utf-8"))
        assert report["stats"]["succeeded"] == 2
        assert report["stats"]["failed"] == 0

    def test_resume_skips_already_done(self, tmp_path: Path) -> None:
        prompts_path = tmp_path / "prompts.jsonl"
        _write_prompts(prompts_path, [("a1", "p1"), ("b2", "p2"), ("c3", "p3")])

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Pretend 'a1' was already processed in a previous run.
        (out_dir / "generated_code.jsonl").write_text(
            json.dumps({"prompt_id": "a1", "prompt_text": "p1", "generated_code": "pass", "error": None}) + "\n",
            encoding="utf-8",
        )

        gen = _FakeGenerator(responses={"p2": "```python\nx=2\n```", "p3": "```python\nx=3\n```"})
        runner = CodeGeneratorRunner(gen, _make_config())
        runner.run(prompts_path, out_dir)

        # Only p2 and p3 should have been generated; p1 must NOT have been called.
        assert gen.calls == ["p2", "p3"]
        all_recs = [
            json.loads(line)
            for line in (out_dir / "generated_code.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {r["prompt_id"] for r in all_recs} == {"a1", "b2", "c3"}

    def test_resume_retries_failed_rows(self, tmp_path: Path) -> None:
        """Failed rows from a previous run must be retried, not treated as done."""
        prompts_path = tmp_path / "prompts.jsonl"
        _write_prompts(prompts_path, [("a1", "p1"), ("b2", "p2"), ("c3", "p3")])

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # a1 succeeded last run; b2 + c3 failed (e.g. tunnel was down).
        (out_dir / "generated_code.jsonl").write_text(
            json.dumps({"prompt_id": "a1", "prompt_text": "p1",
                        "generated_code": "x=1", "error": None}) + "\n"
            + json.dumps({"prompt_id": "b2", "prompt_text": "p2",
                          "generated_code": None,
                          "error": "HTTPError: 404"}) + "\n"
            + json.dumps({"prompt_id": "c3", "prompt_text": "p3",
                          "generated_code": None,
                          "error": "HTTPError: 404"}) + "\n",
            encoding="utf-8",
        )

        gen = _FakeGenerator(responses={"p2": "```python\nx=2\n```",
                                        "p3": "```python\nx=3\n```"})
        runner = CodeGeneratorRunner(gen, _make_config())
        runner.run(prompts_path, out_dir)

        # b2 and c3 must be retried; a1 must NOT be re-called.
        assert sorted(gen.calls) == ["p2", "p3"]

        # Main file now has 3 success rows, no duplicate prompt_ids.
        recs = [
            json.loads(line) for line in
            (out_dir / "generated_code.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pids = [r["prompt_id"] for r in recs]
        assert sorted(pids) == ["a1", "b2", "c3"]
        assert all(r["error"] is None for r in recs)

        # Failed rows archived to a sibling file for audit.
        failed_path = out_dir / "generated_code.failed.jsonl"
        assert failed_path.exists()
        failed = [
            json.loads(line) for line in
            failed_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert {f["prompt_id"] for f in failed} == {"b2", "c3"}
        assert all(f["error"] is not None for f in failed)

    def test_per_prompt_failure_does_not_abort_run(self, tmp_path: Path) -> None:
        prompts_path = tmp_path / "prompts.jsonl"
        _write_prompts(prompts_path, [("a1", "ok-prompt"), ("b2", "boom"), ("c3", "ok-prompt-2")])

        gen = _FakeGenerator(
            responses={
                "ok-prompt": "```python\nprint(1)\n```",
                "ok-prompt-2": "```python\nprint(2)\n```",
            },
            failures={"boom"},
        )
        runner = CodeGeneratorRunner(gen, _make_config())
        out_dir = tmp_path / "out"
        runner.run(prompts_path, out_dir)

        recs = [
            json.loads(line)
            for line in (out_dir / "generated_code.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(recs) == 3
        bad = next(r for r in recs if r["prompt_id"] == "b2")
        assert bad["error"] is not None
        assert bad["generated_code"] is None
        assert not (out_dir / "code_files" / "b2.py").exists()
        # Good prompts still got their .py files.
        assert (out_dir / "code_files" / "a1.py").exists()
        assert (out_dir / "code_files" / "c3.py").exists()

        report = json.loads((out_dir / "code_generator_report.json").read_text(encoding="utf-8"))
        assert report["stats"]["succeeded"] == 2
        assert report["stats"]["failed"] == 1

    def test_missing_prompts_file_raises(self, tmp_path: Path) -> None:
        runner = CodeGeneratorRunner(_FakeGenerator(responses={}), _make_config())
        with pytest.raises(FileNotFoundError):
            runner.run(tmp_path / "nope.jsonl", tmp_path / "out")
