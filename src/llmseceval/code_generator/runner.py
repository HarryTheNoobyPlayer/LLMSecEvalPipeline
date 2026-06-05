"""Drives a code generator over a JSONL of prompts, with crash-safe resume.

Reads ``prompts_clean.jsonl`` produced by the Data Loader, sends each prompt
to a ``BaseCodeGenerator``, and appends one record per prompt to
``generated_code.jsonl``.  Each row is flushed immediately so an interrupted
run can be resumed by simply re-invoking the runner: already-processed
``prompt_id`` values are read from the existing output file and skipped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from llmseceval.code_generator.base import BaseCodeGenerator
from llmseceval.code_generator.extractor import extract_code
from llmseceval.config import CodeGeneratorConfig

logger = logging.getLogger(__name__)


class CodeGeneratorRunner:
    """Iterate over prompts, generate code, and persist results incrementally."""

    def __init__(
        self,
        generator: BaseCodeGenerator,
        config: CodeGeneratorConfig,
    ) -> None:
        self.generator = generator
        self.config = config

    # ---- public API -------------------------------------------------------

    def run(
        self,
        prompts_jsonl: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Generate code for every prompt in *prompts_jsonl*.

        Parameters
        ----------
        prompts_jsonl:
            Path to ``prompts_clean.jsonl`` produced by the Data Loader.
        output_dir:
            Directory to write ``generated_code.jsonl``, ``code_files/``,
            and ``code_generator_report.json`` into.

        Returns
        -------
        Path
            Path to the resulting ``generated_code.jsonl``.
        """
        prompts_jsonl = Path(prompts_jsonl)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_jsonl = output_dir / "generated_code.jsonl"
        code_dir = output_dir / "code_files"
        code_dir.mkdir(parents=True, exist_ok=True)

        all_prompts = list(self._read_prompts(prompts_jsonl))
        done_ids = self._prepare_resume(output_jsonl)
        todo = [p for p in all_prompts if p["prompt_id"] not in done_ids]

        logger.info(
            "Code generation: %d prompts total | %d already done | %d to do",
            len(all_prompts),
            len(done_ids),
            len(todo),
        )

        stats: dict[str, Any] = {
            "total_prompts": len(all_prompts),
            "already_done": len(done_ids),
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "total_tokens": 0,
            "total_generation_time_s": 0.0,
        }

        model_info = self.generator.get_model_info()
        t_run_start = time.perf_counter()

        with open(output_jsonl, "a", encoding="utf-8") as out_fh:
            with Progress(
                TextColumn("[bold blue]Generating code"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Generating...", total=len(todo))
                for prompt in todo:
                    record = self._process_one(prompt, model_info)
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()

                    if record["error"] is None and record["generated_code"]:
                        (code_dir / f"{prompt['prompt_id']}.py").write_text(
                            record["generated_code"],
                            encoding="utf-8",
                        )

                    stats["attempted"] += 1
                    if record["error"]:
                        stats["failed"] += 1
                    else:
                        stats["succeeded"] += 1
                        if record.get("token_count"):
                            stats["total_tokens"] += record["token_count"]
                        if record.get("generation_time_s"):
                            stats["total_generation_time_s"] += record["generation_time_s"]
                    progress.advance(task)

        stats["wall_time_s"] = round(time.perf_counter() - t_run_start, 2)
        if stats["succeeded"]:
            stats["avg_tokens"] = round(stats["total_tokens"] / stats["succeeded"], 1)
            stats["avg_generation_time_s"] = round(
                stats["total_generation_time_s"] / stats["succeeded"], 3
            )
        else:
            stats["avg_tokens"] = 0
            stats["avg_generation_time_s"] = 0.0

        report_path = output_dir / "code_generator_report.json"
        report_path.write_text(
            json.dumps({"model": model_info, "stats": stats}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Code generation complete, %d ok / %d fail in %.1fs (report: %s)",
            stats["succeeded"],
            stats["failed"],
            stats["wall_time_s"],
            report_path,
        )
        return output_jsonl

    # ---- internals --------------------------------------------------------

    def _process_one(
        self,
        prompt: dict[str, Any],
        model_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate code for a single prompt and assemble the output record."""
        prompt_id = prompt["prompt_id"]
        prompt_text = prompt["text"]

        try:
            result = self.generator.generate(prompt_text)
            code = extract_code(result.raw_response)
            return {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "generated_code": code,
                "raw_response": result.raw_response,
                "thinking": result.thinking,
                "model_name": model_info["model_name"],
                "generation_params": model_info["generation_params"],
                "token_count": result.token_count,
                "generation_time_s": result.generation_time_s,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001, per-prompt errors must not abort run
            logger.warning("Generation failed for %s: %s", prompt_id, exc)
            return {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "generated_code": None,
                "raw_response": None,
                "thinking": None,
                "model_name": model_info["model_name"],
                "generation_params": model_info["generation_params"],
                "token_count": None,
                "generation_time_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _read_prompts(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _prepare_resume(output_jsonl: Path) -> set[str]:
        """Read existing output, archive failed rows, return successful IDs.

        Why this is non-trivial: a mass-failure scenario (tunnel down →
        every prompt returns an HTTP error) used to write a row per prompt
        with ``error`` set, and the previous "any prompt_id present means
        done" logic then blocked all retries.  Now we:

        - Treat only rows with ``error is None`` as done.
        - Move failed rows to a sibling ``<name>.failed.jsonl`` so the run's
          failure audit trail is preserved without polluting the main file.
        - Atomically rewrite the main file with only successful rows so
          retries don't accumulate duplicate ``prompt_id`` rows.
        """
        if not output_jsonl.exists():
            return set()

        success_lines: list[str] = []
        failed_lines: list[str] = []
        success_ids: set[str] = set()
        with open(output_jsonl, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw if raw.endswith("\n") else raw + "\n"
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Dropping malformed line in %s", output_jsonl)
                    continue
                pid = rec.get("prompt_id")
                if not pid:
                    continue
                if rec.get("error") is None:
                    success_lines.append(line)
                    success_ids.add(pid)
                else:
                    failed_lines.append(line)

        if not failed_lines:
            return success_ids

        # Archive failures, then atomically rewrite the main file.
        failed_path = output_jsonl.with_name(output_jsonl.stem + ".failed.jsonl")
        with open(failed_path, "a", encoding="utf-8") as fh:
            fh.writelines(failed_lines)

        tmp_path = output_jsonl.with_name(output_jsonl.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.writelines(success_lines)
        tmp_path.replace(output_jsonl)

        logger.info(
            "Resume: %d ok + %d failed in %s; archived failures to %s, "
            "will retry %d prompt(s).",
            len(success_ids), len(failed_lines), output_jsonl.name,
            failed_path.name, len(failed_lines),
        )
        return success_ids


# ---------------------------------------------------------------------------
# Standalone entry point (used until cli.py exists)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from llmseceval.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    prompts_path = sys.argv[2] if len(sys.argv) > 2 else "./results/data_loader_output/prompts_clean.jsonl"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./results/code_generator_output"

    cfg = load_config(config_path)

    if cfg.code_generator.backend == "lmstudio":
        from llmseceval.code_generator.lmstudio_generator import LMStudioGenerator
        generator = LMStudioGenerator(cfg.code_generator)
    else:
        from llmseceval.code_generator.ollama_generator import OllamaGenerator
        generator = OllamaGenerator(cfg.code_generator)

    runner = CodeGeneratorRunner(generator, cfg.code_generator)
    runner.run(prompts_jsonl=prompts_path, output_dir=output_dir)
