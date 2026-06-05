"""Abstract base class for SAST executors.

Any tool (Bandit, Semgrep, ...) implements this interface so the runner
stays tool-agnostic.  The dataclasses below capture the normalised shape
the rest of the pipeline consumes, concrete executors are responsible
for translating their tool's native output into ``SASTFinding`` rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SASTFinding:
    """One vulnerability flagged by a SAST tool.

    Attributes
    ----------
    test_id:
        Tool-specific check identifier (e.g. Bandit's ``B303``).
    test_name:
        Tool-specific check name (e.g. ``blacklist``).
    severity:
        ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"``, normalised to uppercase.
    confidence:
        ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"``, normalised to uppercase.
    cwe_id:
        CWE identifier if the tool reports one, else ``None``.
    line_number:
        1-indexed line where the issue starts.
    line_range:
        Full range of lines the issue spans.
    issue_text:
        Human-readable description.
    code_snippet:
        The offending lines (truncated to 500 chars).
    more_info:
        Optional URL to vendor documentation for the check.
    """

    test_id: str
    test_name: str
    severity: str
    confidence: str
    cwe_id: int | None
    line_number: int
    line_range: list[int]
    issue_text: str
    code_snippet: str
    more_info: str | None = None


@dataclass
class SASTResult:
    """All output from scanning a single file.

    Attributes
    ----------
    findings:
        Normalised list of findings (may be empty).
    bandit_errors:
        Tool-reported errors (e.g. syntax errors that prevented analysis).
        Kept separate from runner-level errors.
    exit_code:
        Tool process exit code.  For Bandit: ``0`` = no issues, ``1`` =
        issues found, ``2`` = internal error.  ``-1`` is used for timeouts.
    scan_time_s:
        Wall-clock scan time in seconds.
    raw_output:
        The tool's full structured output (parsed JSON), for debugging.
    """

    findings: list[SASTFinding] = field(default_factory=list)
    bandit_errors: list[dict[str, Any]] = field(default_factory=list)
    exit_code: int = 0
    scan_time_s: float = 0.0
    raw_output: dict[str, Any] | None = None


class BaseSASTExecutor(ABC):
    """Minimal interface every SAST tool wrapper must implement."""

    @abstractmethod
    def scan(self, file_path: Path) -> SASTResult:
        """Scan *file_path* and return a normalised :class:`SASTResult`.

        Implementations must not raise on tool-reported issues, those go
        into ``bandit_errors``.  They MAY raise on transport-level failures
        (e.g. the tool binary is missing); the runner catches and records
        such failures per-file so a single bad file never halts the run.
        """

    @abstractmethod
    def get_tool_info(self) -> dict[str, Any]:
        """Return metadata identifying the tool, its version, and config.

        The returned dict is embedded into the run report so results files
        are self-describing.
        """
