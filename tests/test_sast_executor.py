"""Unit tests for the SAST executor stage.

Covers:
- ``BanditExecutor`` request shaping & JSON parsing (with mocked subprocess)
- ``SASTExecutorRunner`` checkpoint / resume / per-prompt error isolation /
  missing code-file handling.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llmseceval.config import BanditConfig, SASTExecutorConfig
from llmseceval.sast_executor import (
    BanditExecutor,
    BaseSASTExecutor,
    SASTExecutorRunner,
    SASTFinding,
    SASTResult,
)


# ===========================================================================
# BanditExecutor, subprocess mocked
# ===========================================================================


def _make_sast_config(**overrides: Any) -> SASTExecutorConfig:
    bandit_overrides = overrides.pop("bandit", {})
    base = SASTExecutorConfig(
        tool="bandit",
        bandit=BanditConfig(
            severity_threshold=bandit_overrides.get("severity_threshold", "LOW"),
            confidence_threshold=bandit_overrides.get("confidence_threshold", "LOW"),
            extra_args=bandit_overrides.get("extra_args", []),
        ),
        timeout_per_file_s=overrides.get("timeout_per_file_s", 5),
        parallel_workers=overrides.get("parallel_workers", 2),
    )
    return base


_SAMPLE_BANDIT_JSON = {
    "errors": [],
    "generated_at": "2026-05-20T00:00:00Z",
    "metrics": {"_totals": {"loc": 30}},
    "results": [
        {
            "code": "subprocess.check_output(['ps', 'aux'])",
            "filename": "x.py",
            "issue_confidence": "HIGH",
            "issue_cwe": {"id": 78, "link": "https://cwe.mitre.org/data/definitions/78.html"},
            "issue_severity": "LOW",
            "issue_text": "Consider possible security implications associated with the subprocess module.",
            "line_number": 5,
            "line_range": [5],
            "more_info": "https://bandit.readthedocs.io/en/.../b404.html",
            "test_id": "B404",
            "test_name": "blacklist",
        },
        {
            "code": "hashlib.md5(b'data').hexdigest()",
            "filename": "x.py",
            "issue_confidence": "HIGH",
            "issue_cwe": {"id": 327, "link": "..."},
            "issue_severity": "MEDIUM",
            "issue_text": "Use of weak MD5 hash for security.",
            "line_number": 10,
            "line_range": [10],
            "more_info": "https://bandit.readthedocs.io/en/.../b303.html",
            "test_id": "B303",
            "test_name": "blacklist",
        },
    ],
}


class TestBanditExecutor:
    def test_command_shape(self) -> None:
        cfg = _make_sast_config(bandit={"severity_threshold": "MEDIUM",
                                        "confidence_threshold": "HIGH",
                                        "extra_args": ["--skip", "B101"]})
        ex = BanditExecutor(cfg)
        assert ex._cmd_base == [
            "bandit", "-f", "json", "-q",
            "--severity-level", "medium",
            "--confidence-level", "high",
            "--skip", "B101",
        ]

    def test_invalid_threshold_raises(self) -> None:
        cfg = _make_sast_config(bandit={"severity_threshold": "EXTREME"})
        with pytest.raises(ValueError, match="severity_threshold"):
            BanditExecutor(cfg)

    def test_scan_parses_findings(self, tmp_path: Path) -> None:
        ex = BanditExecutor(_make_sast_config())
        target = tmp_path / "x.py"
        target.write_text("pass\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps(_SAMPLE_BANDIT_JSON)
        mock_proc.stderr = ""
        mock_proc.returncode = 1  # bandit returns 1 when issues found

        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            result = ex.scan(target)

        # Subprocess called with full command + target path.
        called_args = mock_run.call_args.args[0]
        assert called_args[:5] == ["bandit", "-f", "json", "-q", "--severity-level"]
        assert called_args[-1] == str(target)

        assert result.exit_code == 1
        assert len(result.findings) == 2
        f0, f1 = result.findings
        assert f0.test_id == "B404"
        assert f0.severity == "LOW"
        assert f0.confidence == "HIGH"
        assert f0.cwe_id == 78
        assert f0.line_number == 5
        assert f1.test_id == "B303"
        assert f1.cwe_id == 327
        assert f1.severity == "MEDIUM"
        assert result.bandit_errors == []
        assert result.raw_output == _SAMPLE_BANDIT_JSON

    def test_scan_handles_clean_file(self, tmp_path: Path) -> None:
        """No findings, exit code 0."""
        ex = BanditExecutor(_make_sast_config())
        target = tmp_path / "clean.py"
        target.write_text("x = 1\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps({"errors": [], "results": []})
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            result = ex.scan(target)
        assert result.exit_code == 0
        assert result.findings == []
        assert result.bandit_errors == []

    def test_scan_handles_timeout(self, tmp_path: Path) -> None:
        ex = BanditExecutor(_make_sast_config(timeout_per_file_s=2))
        target = tmp_path / "slow.py"
        target.write_text("pass\n", encoding="utf-8")

        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="bandit", timeout=2)):
            result = ex.scan(target)
        assert result.exit_code == -1
        assert result.findings == []
        assert result.bandit_errors[0]["reason"] == "timeout"
        assert result.scan_time_s == 2.0

    def test_scan_handles_bad_json(self, tmp_path: Path) -> None:
        ex = BanditExecutor(_make_sast_config())
        target = tmp_path / "x.py"
        target.write_text("pass\n", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.stdout = "not-json"
        mock_proc.stderr = "bandit crashed"
        mock_proc.returncode = 2

        with patch("subprocess.run", return_value=mock_proc):
            result = ex.scan(target)
        assert result.exit_code == 2
        assert result.findings == []
        assert result.bandit_errors[0]["reason"] == "json_decode_error"

    def test_scan_handles_missing_cwe(self, tmp_path: Path) -> None:
        """Older Bandit versions may omit issue_cwe, cwe_id should be None."""
        ex = BanditExecutor(_make_sast_config())
        target = tmp_path / "x.py"
        target.write_text("pass\n", encoding="utf-8")

        no_cwe = {
            "errors": [], "results": [{
                "code": "x",
                "filename": "x.py",
                "issue_confidence": "LOW",
                "issue_severity": "LOW",
                "issue_text": "...",
                "line_number": 1,
                "line_range": [1],
                "test_id": "B999",
                "test_name": "fake",
            }],
        }
        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps(no_cwe)
        mock_proc.stderr = ""
        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            result = ex.scan(target)
        assert result.findings[0].cwe_id is None

    def test_get_tool_info(self) -> None:
        ex = BanditExecutor(_make_sast_config())
        mock_proc = MagicMock()
        mock_proc.stdout = "bandit 1.9.4\n  python version = 3.12"
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            info = ex.get_tool_info()
        assert info["tool"] == "bandit"
        assert "bandit 1.9.4" in info["version"]
        assert info["command"][0] == "bandit"
        assert info["config"]["severity_threshold"] == "LOW"


# ===========================================================================
# SASTExecutorRunner, with a fake executor
# ===========================================================================


class _FakeExecutor(BaseSASTExecutor):
    """Returns canned SASTResults keyed by file basename."""

    def __init__(
        self,
        results: dict[str, SASTResult],
        failures: set[str] | None = None,
    ) -> None:
        self.results = results
        self.failures = failures or set()
        self.calls: list[str] = []

    def scan(self, file_path: Path) -> SASTResult:
        stem = file_path.stem
        self.calls.append(stem)
        if stem in self.failures:
            raise RuntimeError(f"simulated SAST failure for {stem}")
        return self.results.get(stem, SASTResult())

    def get_tool_info(self) -> dict[str, Any]:
        return {"tool": "fake", "version": "0.0", "command": ["fake"], "config": {}}


def _write_generated(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_finding(test_id: str = "B303", severity: str = "MEDIUM",
                  cwe_id: int | None = 327) -> SASTFinding:
    return SASTFinding(
        test_id=test_id,
        test_name="fake",
        severity=severity,
        confidence="HIGH",
        cwe_id=cwe_id,
        line_number=1,
        line_range=[1],
        issue_text="...",
        code_snippet="x",
        more_info=None,
    )


class TestSASTRunner:
    def test_happy_path(self, tmp_path: Path) -> None:
        gen_path = tmp_path / "generated_code.jsonl"
        _write_generated(gen_path, [
            {"prompt_id": "a1", "generated_code": "import os\n", "error": None},
            {"prompt_id": "b2", "generated_code": "import md5\n", "error": None},
        ])
        code_dir = tmp_path / "code_files"
        code_dir.mkdir()
        (code_dir / "a1.py").write_text("import os\n")
        (code_dir / "b2.py").write_text("import md5\n")

        fake = _FakeExecutor({
            "a1": SASTResult(findings=[], exit_code=0, scan_time_s=0.1),
            "b2": SASTResult(
                findings=[_make_finding("B303", "MEDIUM", 327),
                          _make_finding("B404", "LOW", 78)],
                exit_code=1,
                scan_time_s=0.2,
            ),
        })
        cfg = _make_sast_config(parallel_workers=1)
        runner = SASTExecutorRunner(fake, cfg)
        out_dir = tmp_path / "out"
        out_jsonl = runner.run(gen_path, out_dir)

        recs = [json.loads(line) for line in
                out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {r["prompt_id"] for r in recs} == {"a1", "b2"}
        rec_b = next(r for r in recs if r["prompt_id"] == "b2")
        assert rec_b["findings_count"] == 2
        assert rec_b["skipped"] is False
        assert rec_b["error"] is None
        assert rec_b["findings"][0]["test_id"] in {"B303", "B404"}

        report = json.loads((out_dir / "sast_executor_report.json")
                            .read_text(encoding="utf-8"))
        assert report["stats"]["scanned"] == 2
        assert report["stats"]["findings_total"] == 2
        assert report["stats"]["by_severity"] == {"MEDIUM": 1, "LOW": 1}
        assert report["stats"]["by_test_id"] == {"B303": 1, "B404": 1}
        assert report["stats"]["by_cwe"] == {"327": 1, "78": 1}
        assert report["stats"]["prompts_with_any_finding"] == 1
        assert report["stats"]["prompts_with_no_findings"] == 1

    def test_resume_skips_done(self, tmp_path: Path) -> None:
        gen_path = tmp_path / "generated_code.jsonl"
        _write_generated(gen_path, [
            {"prompt_id": "a1", "generated_code": "x", "error": None},
            {"prompt_id": "b2", "generated_code": "x", "error": None},
            {"prompt_id": "c3", "generated_code": "x", "error": None},
        ])
        code_dir = tmp_path / "code_files"
        code_dir.mkdir()
        for pid in ("a1", "b2", "c3"):
            (code_dir / f"{pid}.py").write_text("x = 1\n")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "sast_findings.jsonl").write_text(json.dumps({
            "prompt_id": "a1", "skipped": False, "findings_count": 0,
            "findings": [], "bandit_errors": [], "exit_code": 0,
            "scan_time_s": 0.0, "error": None,
        }) + "\n", encoding="utf-8")

        fake = _FakeExecutor({"b2": SASTResult(), "c3": SASTResult()})
        runner = SASTExecutorRunner(fake, _make_sast_config(parallel_workers=1))
        runner.run(gen_path, out_dir)

        assert set(fake.calls) == {"b2", "c3"}  # a1 NOT re-scanned
        recs = [json.loads(line) for line in
                (out_dir / "sast_findings.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {r["prompt_id"] for r in recs} == {"a1", "b2", "c3"}

    def test_resume_retries_failed_rows(self, tmp_path: Path) -> None:
        """SAST-side errored rows from a previous run must be retried."""
        gen_path = tmp_path / "generated_code.jsonl"
        _write_generated(gen_path, [
            {"prompt_id": "ok", "generated_code": "x", "error": None},
            {"prompt_id": "retry_me", "generated_code": "x", "error": None},
        ])
        code_dir = tmp_path / "code_files"
        code_dir.mkdir()
        (code_dir / "ok.py").write_text("x = 1\n")
        (code_dir / "retry_me.py").write_text("x = 1\n")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # 'ok' was scanned cleanly last run; 'retry_me' errored (e.g. bandit crashed).
        (out_dir / "sast_findings.jsonl").write_text(
            json.dumps({"prompt_id": "ok", "skipped": False, "findings_count": 0,
                        "findings": [], "bandit_errors": [], "exit_code": 0,
                        "scan_time_s": 0.1, "error": None}) + "\n"
            + json.dumps({"prompt_id": "retry_me", "skipped": False,
                          "findings_count": 0, "findings": [],
                          "bandit_errors": [], "exit_code": None,
                          "scan_time_s": None,
                          "error": "RuntimeError: oops"}) + "\n",
            encoding="utf-8",
        )

        fake = _FakeExecutor({"retry_me": SASTResult(exit_code=0, scan_time_s=0.1)})
        runner = SASTExecutorRunner(fake, _make_sast_config(parallel_workers=1))
        runner.run(gen_path, out_dir)

        # Only retry_me re-scanned; ok stayed put.
        assert fake.calls == ["retry_me"]

        recs = [
            json.loads(line) for line in
            (out_dir / "sast_findings.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pids = [r["prompt_id"] for r in recs]
        assert sorted(pids) == ["ok", "retry_me"]
        assert all(r["error"] is None for r in recs)

        failed_path = out_dir / "sast_findings.failed.jsonl"
        assert failed_path.exists()
        failed = [
            json.loads(line) for line in
            failed_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert {f["prompt_id"] for f in failed} == {"retry_me"}

    def test_per_file_failure_isolated(self, tmp_path: Path) -> None:
        gen_path = tmp_path / "generated_code.jsonl"
        _write_generated(gen_path, [
            {"prompt_id": "ok", "generated_code": "x", "error": None},
            {"prompt_id": "bad", "generated_code": "x", "error": None},
        ])
        code_dir = tmp_path / "code_files"
        code_dir.mkdir()
        (code_dir / "ok.py").write_text("x = 1\n")
        (code_dir / "bad.py").write_text("x = 1\n")

        fake = _FakeExecutor({"ok": SASTResult()}, failures={"bad"})
        runner = SASTExecutorRunner(fake, _make_sast_config(parallel_workers=1))
        out_dir = tmp_path / "out"
        runner.run(gen_path, out_dir)

        recs = {json.loads(line)["prompt_id"]: json.loads(line)
                for line in (out_dir / "sast_findings.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()}
        assert recs["ok"]["error"] is None
        assert recs["bad"]["error"] is not None
        assert "simulated SAST failure" in recs["bad"]["error"]

        report = json.loads((out_dir / "sast_executor_report.json")
                            .read_text(encoding="utf-8"))
        assert report["stats"]["scanned"] == 1
        assert report["stats"]["errored"] == 1

    def test_skips_failed_generation(self, tmp_path: Path) -> None:
        gen_path = tmp_path / "generated_code.jsonl"
        _write_generated(gen_path, [
            {"prompt_id": "had_error",
             "generated_code": None, "error": "BackendDown: oops"},
            {"prompt_id": "no_code",
             "generated_code": "", "error": None},
            {"prompt_id": "missing_file",
             "generated_code": "x", "error": None},
        ])
        # Note: no .py files written → code_files dir is empty
        (tmp_path / "code_files").mkdir()

        fake = _FakeExecutor({})
        runner = SASTExecutorRunner(fake, _make_sast_config(parallel_workers=1))
        out_dir = tmp_path / "out"
        runner.run(gen_path, out_dir)

        recs = {json.loads(line)["prompt_id"]: json.loads(line)
                for line in (out_dir / "sast_findings.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()}
        assert recs["had_error"]["skip_reason"] == "code_generation_error"
        assert recs["no_code"]["skip_reason"] == "empty_generated_code"
        assert recs["missing_file"]["skip_reason"] == "code_file_missing"
        assert fake.calls == []  # no scans attempted

        report = json.loads((out_dir / "sast_executor_report.json")
                            .read_text(encoding="utf-8"))
        assert report["stats"]["skipped"] == 3
        assert report["stats"]["scanned"] == 0

    def test_parallel_execution(self, tmp_path: Path) -> None:
        """All prompts scanned regardless of worker count; order doesn't matter."""
        gen_path = tmp_path / "generated_code.jsonl"
        rows = [
            {"prompt_id": f"p{i:02d}", "generated_code": "x", "error": None}
            for i in range(8)
        ]
        _write_generated(gen_path, rows)
        code_dir = tmp_path / "code_files"
        code_dir.mkdir()
        for r in rows:
            (code_dir / f"{r['prompt_id']}.py").write_text("x = 1\n")

        fake = _FakeExecutor({r["prompt_id"]: SASTResult() for r in rows})
        runner = SASTExecutorRunner(fake, _make_sast_config(parallel_workers=4))
        out_dir = tmp_path / "out"
        runner.run(gen_path, out_dir)

        recs = [json.loads(line) for line in
                (out_dir / "sast_findings.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {r["prompt_id"] for r in recs} == {f"p{i:02d}" for i in range(8)}
        assert set(fake.calls) == {f"p{i:02d}" for i in range(8)}

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        fake = _FakeExecutor({})
        runner = SASTExecutorRunner(fake, _make_sast_config())
        with pytest.raises(FileNotFoundError):
            runner.run(tmp_path / "nope.jsonl", tmp_path / "out")
