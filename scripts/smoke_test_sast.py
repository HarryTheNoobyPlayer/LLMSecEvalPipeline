"""Smoke test for the SAST executor.

Runs Bandit against the .py files produced by ``scripts/smoke_test_codegen.py``
(``results/codegen_smoke/``) and pretty-prints per-prompt findings.

Usage:
    python scripts/smoke_test_sast.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmseceval.config import load_config
from llmseceval.sast_executor import BanditExecutor, SASTExecutorRunner

ROOT = Path(__file__).parent.parent
CODEGEN_DIR = ROOT / "results" / "codegen_smoke"
GEN_JSONL = CODEGEN_DIR / "generated_code.jsonl"
SMOKE_DIR = ROOT / "results" / "sast_smoke"


def print_per_prompt(jsonl_path: Path, console: Console) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    for i, rec in enumerate(records, 1):
        header = f"[bold cyan]#{i}  prompt_id={rec['prompt_id']}[/]"
        if rec["skipped"]:
            header += f" | [yellow]SKIPPED ({rec['skip_reason']})[/]"
            console.rule(header)
            continue
        if rec["error"]:
            header += f" | [red]ERROR: {rec['error']}[/]"
            console.rule(header)
            continue

        header += (
            f" | findings={rec['findings_count']}"
            f"  exit={rec['exit_code']}"
            f"  time={rec.get('scan_time_s', 0):.2f}s"
        )
        console.rule(header)

        if not rec["findings"]:
            console.print(Panel("(no findings)", border_style="green"))
            continue

        table = Table(show_lines=False, border_style="red")
        table.add_column("Line", justify="right")
        table.add_column("Test")
        table.add_column("Sev")
        table.add_column("Conf")
        table.add_column("CWE")
        table.add_column("Issue", overflow="fold")
        for f in rec["findings"]:
            table.add_row(
                str(f["line_number"]),
                f["test_id"],
                f["severity"],
                f["confidence"],
                str(f.get("cwe_id") or "-"),
                f["issue_text"][:80],
            )
        console.print(table)


def print_report_summary(report_path: Path, console: Console) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stats = report["stats"]
    tool = report["tool"]

    console.rule("[bold green]Aggregate Report")
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("tool", f"{tool['tool']} ({tool['version']})")
    summary.add_row("total prompts", str(stats["total_prompts"]))
    summary.add_row("scanned", str(stats["scanned"]))
    summary.add_row("skipped", str(stats["skipped"]))
    summary.add_row("errored", str(stats["errored"]))
    summary.add_row("findings total", str(stats["findings_total"]))
    summary.add_row("prompts with findings", str(stats["prompts_with_any_finding"]))
    summary.add_row("by severity", json.dumps(stats.get("by_severity", {})))
    summary.add_row("by test_id (top)", json.dumps(stats.get("by_test_id", {})))
    summary.add_row("by cwe (top)", json.dumps(stats.get("by_cwe", {})))
    summary.add_row("wall time", f"{stats['wall_time_s']}s")
    console.print(summary)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    console = Console()

    if not GEN_JSONL.exists():
        console.print(f"[red]No codegen smoke output at {GEN_JSONL}[/]")
        console.print("Run [bold]scripts/smoke_test_codegen.py[/] first.")
        return 1

    cfg = load_config(ROOT / "config.yaml")
    console.print(
        f"[bold]Tool:[/] {cfg.sast_executor.tool}  "
        f"[bold]severity≥[/]{cfg.sast_executor.bandit.severity_threshold}  "
        f"[bold]confidence≥[/]{cfg.sast_executor.bandit.confidence_threshold}  "
        f"[bold]workers=[/]{cfg.sast_executor.parallel_workers}",
    )

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    executor = BanditExecutor(cfg.sast_executor)
    runner = SASTExecutorRunner(executor, cfg.sast_executor)
    out_jsonl = runner.run(
        generated_code_jsonl=GEN_JSONL,
        output_dir=SMOKE_DIR,
        code_files_dir=CODEGEN_DIR / "code_files",
    )

    console.print()
    print_per_prompt(out_jsonl, console)
    print_report_summary(SMOKE_DIR / "sast_executor_report.json", console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
