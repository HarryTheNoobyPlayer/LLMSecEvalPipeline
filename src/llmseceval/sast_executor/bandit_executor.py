"""Bandit-backed SAST executor.

Runs ``bandit -f json -q`` as a subprocess per file and normalises the
output into :class:`SASTResult` / :class:`SASTFinding` rows.  Subprocess
isolation lets us cleanly timeout a single file without affecting others.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from llmseceval.config import SASTExecutorConfig
from llmseceval.sast_executor.base import BaseSASTExecutor, SASTFinding, SASTResult

logger = logging.getLogger(__name__)

_SEVERITY_LEVELS = {"LOW", "MEDIUM", "HIGH"}
_CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}


class BanditExecutor(BaseSASTExecutor):
    """Run Bandit on individual files via subprocess."""

    def __init__(self, config: SASTExecutorConfig) -> None:
        sev = config.bandit.severity_threshold.upper()
        conf = config.bandit.confidence_threshold.upper()
        if sev not in _SEVERITY_LEVELS:
            raise ValueError(f"Invalid severity_threshold: {sev}")
        if conf not in _CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence_threshold: {conf}")

        self.config = config
        self._cmd_base = self._build_command_base()

    def _build_command_base(self) -> list[str]:
        cmd = [
            "bandit",
            "-f", "json",
            "-q",
            "--severity-level", self.config.bandit.severity_threshold.lower(),
            "--confidence-level", self.config.bandit.confidence_threshold.lower(),
        ]
        cmd.extend(self.config.bandit.extra_args)
        return cmd

    def scan(self, file_path: Path) -> SASTResult:
        file_path = Path(file_path)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [*self._cmd_base, str(file_path)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_file_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Bandit timed out on %s after %ds",
                           file_path, self.config.timeout_per_file_s)
            return SASTResult(
                findings=[],
                bandit_errors=[{"reason": "timeout",
                                "timeout_s": self.config.timeout_per_file_s}],
                exit_code=-1,
                scan_time_s=float(self.config.timeout_per_file_s),
                raw_output=None,
            )

        scan_time = round(time.perf_counter() - t0, 3)
        data: dict[str, Any] = {}
        if proc.stdout.strip():
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse bandit JSON for %s: %s",
                               file_path, exc)
                return SASTResult(
                    findings=[],
                    bandit_errors=[{"reason": "json_decode_error",
                                    "stderr": proc.stderr[:1000]}],
                    exit_code=proc.returncode,
                    scan_time_s=scan_time,
                    raw_output=None,
                )

        findings = [self._parse_finding(r) for r in data.get("results", [])]
        return SASTResult(
            findings=findings,
            bandit_errors=list(data.get("errors", [])),
            exit_code=proc.returncode,
            scan_time_s=scan_time,
            raw_output=data,
        )

    @staticmethod
    def _parse_finding(r: dict[str, Any]) -> SASTFinding:
        cwe = r.get("issue_cwe") or {}
        cwe_id: int | None
        if isinstance(cwe, dict):
            raw = cwe.get("id")
            cwe_id = int(raw) if raw is not None else None
        else:
            cwe_id = None
        return SASTFinding(
            test_id=r.get("test_id", ""),
            test_name=r.get("test_name", ""),
            severity=str(r.get("issue_severity", "")).upper(),
            confidence=str(r.get("issue_confidence", "")).upper(),
            cwe_id=cwe_id,
            line_number=int(r.get("line_number", 0)),
            line_range=list(r.get("line_range", [])),
            issue_text=r.get("issue_text", ""),
            code_snippet=str(r.get("code", ""))[:500],
            more_info=r.get("more_info"),
        )

    def get_tool_info(self) -> dict[str, Any]:
        version_str = ""
        try:
            ver = subprocess.run(
                ["bandit", "--version"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            version_str = (ver.stdout.strip() or ver.stderr.strip()).splitlines()[0] \
                if (ver.stdout or ver.stderr) else ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Could not query bandit version: %s", exc)
        return {
            "tool": "bandit",
            "version": version_str,
            "command": list(self._cmd_base),
            "config": {
                "severity_threshold": self.config.bandit.severity_threshold,
                "confidence_threshold": self.config.bandit.confidence_threshold,
                "extra_args": list(self.config.bandit.extra_args),
                "timeout_per_file_s": self.config.timeout_per_file_s,
                "parallel_workers": self.config.parallel_workers,
            },
        }
