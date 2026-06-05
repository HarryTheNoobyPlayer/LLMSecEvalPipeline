"""Drives a SAST executor over generated code files, with crash-safe resume.

Reads ``generated_code.jsonl`` from the Code Generator stage, looks up
``code_files/<prompt_id>.py`` for each successful generation, scans them
in parallel via ``ThreadPoolExecutor``, and writes one record per prompt
to ``sast_findings.jsonl``.  Each row is flushed immediately so an
interrupted run can be resumed by re-invoking the runner: already-scanned
``prompt_id`` values are read from the existing output file and skipped.

Failed/empty generations are recorded as ``skipped: true`` so the
downstream aggregator can still account for the full population.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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

from llmseceval.config import SASTExecutorConfig
from llmseceval.sast_executor.base import BaseSASTExecutor, SASTResult

logger = logging.getLogger(__name__)


class SASTExecutorRunner:
    """Iterate over generated code files, scan in parallel, persist incrementally."""

    def __init__(self, executor: BaseSASTExecutor, config: SASTExecutorConfig) -> None:
        self.executor = executor
        self.config = config

    # ---- public API -------------------------------------------------------

    def run(
        self,
        generated_code_jsonl: str | Path,
        output_dir: str | Path,
        code_files_dir: str | Path | None = None,
    ) -> Path:
        """Scan every generated .py file referenced by *generated_code_jsonl*.

        Parameters
        ----------
        generated_code_jsonl:
            Path to ``generated_code.jsonl`` produced by the Code Generator.
        output_dir:
            Directory to write ``sast_findings.jsonl`` and
            ``sast_executor_report.json`` into.
        code_files_dir:
            Directory containing the ``<prompt_id>.py`` files.  Defaults to
            ``<generated_code_jsonl>.parent / "code_files"`` to match the
            Code Generator's layout.

        Returns
        -------
        Path
            Path to the resulting ``sast_findings.jsonl``.
        """
        generated_code_jsonl = Path(generated_code_jsonl)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if code_files_dir is None:
            code_files_dir = generated_code_jsonl.parent / "code_files"
        code_files_dir = Path(code_files_dir)

        output_jsonl = output_dir / "sast_findings.jsonl"

        all_records = list(self._read_generated(generated_code_jsonl))
        done_ids = self._prepare_resume(output_jsonl)
        todo = [r for r in all_records if r["prompt_id"] not in done_ids]

        logger.info(
            "SAST scan: %d prompts total | %d already done | %d to do",
            len(all_records),
            len(done_ids),
            len(todo),
        )

        tool_info = self.executor.get_tool_info()
        stats: dict[str, Any] = {
            "total_prompts": len(all_records),
            "already_done": len(done_ids),
            "attempted": 0,
            "scanned": 0,
            "skipped": 0,
            "errored": 0,
            "findings_total": 0,
            "prompts_with_any_finding": 0,
            "prompts_with_no_findings": 0,
            "total_scan_time_s": 0.0,
        }
        sev_counter: Counter[str] = Counter()
        conf_counter: Counter[str] = Counter()
        test_counter: Counter[str] = Counter()
        cwe_counter: Counter[int] = Counter()

        t_start = time.perf_counter()
        with open(output_jsonl, "a", encoding="utf-8") as out_fh:
            with Progress(
                TextColumn("[bold blue]SAST scanning"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Scanning...", total=len(todo))

                workers = max(1, self.config.parallel_workers)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(self._process_one, rec, code_files_dir): rec
                        for rec in todo
                    }
                    for fut in as_completed(futures):
                        record = fut.result()
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_fh.flush()
                        self._update_stats(
                            record, stats,
                            sev_counter, conf_counter, test_counter, cwe_counter,
                        )
                        progress.advance(task)

        stats["wall_time_s"] = round(time.perf_counter() - t_start, 2)
        stats["total_scan_time_s"] = round(stats["total_scan_time_s"], 3)
        stats["by_severity"] = dict(sev_counter)
        stats["by_confidence"] = dict(conf_counter)
        stats["by_test_id"] = dict(test_counter.most_common(20))
        stats["by_cwe"] = {str(k): v for k, v in cwe_counter.most_common(20)}

        report_path = output_dir / "sast_executor_report.json"
        report_path.write_text(
            json.dumps({"tool": tool_info, "stats": stats}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "SAST scan complete, %d scanned / %d skipped / %d errored in %.1fs "
            "(report: %s)",
            stats["scanned"],
            stats["skipped"],
            stats["errored"],
            stats["wall_time_s"],
            report_path,
        )
        return output_jsonl

    # ---- internals --------------------------------------------------------

    def _process_one(
        self,
        gen_rec: dict[str, Any],
        code_files_dir: Path,
    ) -> dict[str, Any]:
        """Scan one file and return the persisted record shape."""
        prompt_id = gen_rec["prompt_id"]
        code_file = code_files_dir / f"{prompt_id}.py"

        # Codegen failure or empty result → no .py file to scan.
        if gen_rec.get("error") is not None:
            return self._skipped(prompt_id, code_file, "code_generation_error")
        if not gen_rec.get("generated_code"):
            return self._skipped(prompt_id, code_file, "empty_generated_code")
        if not code_file.exists():
            return self._skipped(prompt_id, code_file, "code_file_missing")

        try:
            result: SASTResult = self.executor.scan(code_file)
        except Exception as exc:  # noqa: BLE001, per-file errors must not abort run
            logger.warning("SAST scan failed for %s: %s", prompt_id, exc)
            return {
                "prompt_id": prompt_id,
                "code_file": str(code_file),
                "skipped": False,
                "skip_reason": None,
                "findings_count": 0,
                "findings": [],
                "bandit_errors": [],
                "exit_code": None,
                "scan_time_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "prompt_id": prompt_id,
            "code_file": str(code_file),
            "skipped": False,
            "skip_reason": None,
            "findings_count": len(result.findings),
            "findings": [asdict(f) for f in result.findings],
            "bandit_errors": result.bandit_errors,
            "exit_code": result.exit_code,
            "scan_time_s": result.scan_time_s,
            "error": None,
        }

    @staticmethod
    def _skipped(prompt_id: str, code_file: Path, reason: str) -> dict[str, Any]:
        return {
            "prompt_id": prompt_id,
            "code_file": str(code_file),
            "skipped": True,
            "skip_reason": reason,
            "findings_count": 0,
            "findings": [],
            "bandit_errors": [],
            "exit_code": None,
            "scan_time_s": None,
            "error": None,
        }

    @staticmethod
    def _update_stats(
        record: dict[str, Any],
        stats: dict[str, Any],
        sev: Counter[str],
        conf: Counter[str],
        tests: Counter[str],
        cwes: Counter[int],
    ) -> None:
        stats["attempted"] += 1
        if record["error"]:
            stats["errored"] += 1
            return
        if record["skipped"]:
            stats["skipped"] += 1
            return

        stats["scanned"] += 1
        if record.get("scan_time_s"):
            stats["total_scan_time_s"] += record["scan_time_s"]
        findings = record.get("findings", [])
        stats["findings_total"] += len(findings)
        if findings:
            stats["prompts_with_any_finding"] += 1
        else:
            stats["prompts_with_no_findings"] += 1
        for f in findings:
            sev[f["severity"]] += 1
            conf[f["confidence"]] += 1
            tests[f["test_id"]] += 1
            if f.get("cwe_id") is not None:
                cwes[f["cwe_id"]] += 1

    @staticmethod
    def _read_generated(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Generated code file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _prepare_resume(output_jsonl: Path) -> set[str]:
        """Read existing output, archive runner-errored rows, return done IDs.

        Mirrors :meth:`code_generator.runner.CodeGeneratorRunner._prepare_resume`.
        Rows with ``error is None`` are kept (including legitimate
        ``skipped=true`` records, those should not be retried because the
        underlying generation produced no code).  Rows with ``error`` set
        (i.e. the SAST executor itself crashed) are archived to a sibling
        ``<name>.failed.jsonl`` and the main file is rewritten so the rerun
        re-attempts them.
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
    from llmseceval.sast_executor.bandit_executor import BanditExecutor

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    generated_jsonl = (
        sys.argv[2] if len(sys.argv) > 2
        else "./results/code_generator_output/generated_code.jsonl"
    )
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./results/sast_executor_output"

    cfg = load_config(config_path)
    executor = BanditExecutor(cfg.sast_executor)
    runner = SASTExecutorRunner(executor, cfg.sast_executor)
    runner.run(generated_code_jsonl=generated_jsonl, output_dir=output_dir)
